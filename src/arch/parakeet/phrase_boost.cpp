// arch/parakeet/phrase_boost.cpp - see phrase_boost.h.

#include "phrase_boost.h"

#include "transcribe-env.h"
#include "transcribe-log.h"
#include "transcribe-tokenizer.h"

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <unordered_set>

namespace transcribe::parakeet {

namespace {

// Per-node bonus not yet locked in by a completed phrase: what a break
// from this state revokes.
inline float outstanding(float score, float keep) {
    return score - keep;
}

}  // namespace

float PhraseBoost::arc_bonus(int depth) const {
    // Depth scaling: the first token of a phrase is usually a common word
    // piece ("▁C", "▁Lo"), so it gets half the bonus; deeper tokens, which
    // confirm the phrase, get the full weight.
    return depth <= 1 ? first_ * weight_ : weight_;
}

void PhraseBoost::build(const transcribe::Tokenizer & tok, const std::vector<std::string> & phrases, float weight) {
    nodes_.clear();
    n_phrases_ = 0;
    weight_    = weight;
    // Experiment knobs (prototype): TRANSCRIBE_PARAKEET_VOCAB_BOOST_FIRST
    // scales the first-token bonus; ..._REVOKE=1 puts the revocation
    // penalty into the greedy ranking. Off by default: with a single
    // greedy hypothesis the revocation only acts as a penalty on every
    // non-blank token relative to blank, and since only a non-blank token
    // can move the trie state, a long partial match (e.g. "CLAUDE." of
    // "CLAUDE.md") can park the decoder emitting blanks for the rest of
    // the stream. Measured: weight 3 with revocation truncated 2/28 clips.
    // The state bookkeeping (delta()) is the same either way.
    first_     = 0.5f;
    if (const char * e = transcribe::env::str("TRANSCRIBE_PARAKEET_VOCAB_BOOST_FIRST")) {
        first_ = std::strtof(e, nullptr);
    }
    revoke_ = transcribe::env::flag("TRANSCRIBE_PARAKEET_VOCAB_BOOST_REVOKE");
    if (weight <= 0.0f || phrases.empty()) {
        return;
    }
    nodes_.emplace_back();  // root

    std::vector<std::string> variants;
    for (const std::string & p : phrases) {
        if (p.empty()) {
            continue;
        }
        variants.push_back(p);
        if (p[0] >= 'a' && p[0] <= 'z') {
            std::string cap = p;
            cap[0]          = static_cast<char>(cap[0] - 'a' + 'A');
            variants.push_back(cap);
        }
    }

    std::vector<int32_t> ids;
    for (const std::string & v : variants) {
        ids.clear();
        if (tok.encode_sentencepiece_bpe(v, ids, 0, -1, true) != TRANSCRIBE_OK || ids.empty()) {
            log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "parakeet phrase boost: cannot encode \"%s\"; skipped", v.c_str());
            continue;
        }
        if (std::any_of(ids.begin(), ids.end(), [&](int32_t id) { return tok.is_unknown(id) || id < 0; })) {
            log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "parakeet phrase boost: \"%s\" has an unknown piece; skipped",
                    v.c_str());
            continue;
        }
        int32_t s = 0;
        for (const int32_t id : ids) {
            auto it = nodes_[static_cast<size_t>(s)].next.find(id);
            if (it == nodes_[static_cast<size_t>(s)].next.end()) {
                Node n;
                n.depth = nodes_[static_cast<size_t>(s)].depth + 1;
                n.arc   = arc_bonus(n.depth);
                n.score = nodes_[static_cast<size_t>(s)].score + n.arc;
                nodes_.push_back(std::move(n));
                const auto child = static_cast<int32_t>(nodes_.size() - 1);
                nodes_[static_cast<size_t>(s)].next.emplace(id, child);
                s = child;
            } else {
                s = it->second;
            }
        }
        if (!nodes_[static_cast<size_t>(s)].end) {
            nodes_[static_cast<size_t>(s)].end = true;
            ++n_phrases_;
        }
        // Per-phrase dump only on request: a resident caller starts a stream
        // per utterance, and the glossary is often private.
        static const bool dump = std::getenv("TRANSCRIBE_PARAKEET_VOCAB_BOOST_DEBUG") != nullptr;
        if (dump) {
            std::string pieces;
            for (const int32_t id : ids) {
                pieces += tok.token(id) + " ";
            }
            log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "parakeet phrase boost: \"%s\" -> %s", v.c_str(), pieces.c_str());
        }
    }

    // BFS: fail links (longest proper suffix that is a trie path) and
    // `out`, the deepest completed phrase on the trie path to each node
    // (its score is locked in and survives a later break).
    std::deque<int32_t> q;
    for (auto & [tok_id, child] : nodes_[0].next) {
        nodes_[static_cast<size_t>(child)].fail = 0;
        nodes_[static_cast<size_t>(child)].out  = nodes_[static_cast<size_t>(child)].end ? child : -1;
        q.push_back(child);
    }
    while (!q.empty()) {
        const int32_t s = q.front();
        q.pop_front();
        for (auto & [tok_id, child] : nodes_[static_cast<size_t>(s)].next) {
            int32_t f = nodes_[static_cast<size_t>(s)].fail;
            while (true) {
                auto it = nodes_[static_cast<size_t>(f)].next.find(tok_id);
                if (it != nodes_[static_cast<size_t>(f)].next.end() && it->second != child) {
                    nodes_[static_cast<size_t>(child)].fail = it->second;
                    break;
                }
                if (f == 0) {
                    nodes_[static_cast<size_t>(child)].fail = 0;
                    break;
                }
                f = nodes_[static_cast<size_t>(f)].fail;
            }
            Node & c = nodes_[static_cast<size_t>(child)];
            c.out    = c.end ? child : nodes_[static_cast<size_t>(s)].out;
            q.push_back(child);
        }
    }
    log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "parakeet phrase boost: %d phrase variants, %d trie nodes, weight %.2f",
            n_phrases_, static_cast<int>(nodes_.size()), static_cast<double>(weight_));
}

int32_t PhraseBoost::goto_(int32_t s, int32_t t) const {
    while (true) {
        const Node & n  = nodes_[static_cast<size_t>(s)];
        auto         it = n.next.find(t);
        if (it != n.next.end()) {
            return it->second;
        }
        if (s == 0) {
            return 0;
        }
        s = n.fail;
    }
}

float PhraseBoost::delta(int32_t s, int32_t t, int32_t * next) const {
    if (empty()) {
        *next = 0;
        return 0.0f;
    }
    const Node & cur = nodes_[static_cast<size_t>(s)];
    auto         it  = cur.next.find(t);
    if (it != cur.next.end()) {
        *next = it->second;
        return nodes_[static_cast<size_t>(it->second)].arc;
    }
    // Partial match broken: revoke what is not locked in by a completed
    // phrase, then grant the longest suffix match that t extends.
    const float   keep = cur.out >= 0 ? nodes_[static_cast<size_t>(cur.out)].score : 0.0f;
    const int32_t n    = s == 0 ? 0 : goto_(cur.fail, t);
    *next              = n;
    return nodes_[static_cast<size_t>(n)].score - outstanding(cur.score, keep);
}

void PhraseBoost::apply(int32_t s, float * logits, int n, int blank_id) const {
    if (empty()) {
        return;
    }
    const Node & cur    = nodes_[static_cast<size_t>(s)];
    const float  keep   = cur.out >= 0 ? nodes_[static_cast<size_t>(cur.out)].score : 0.0f;
    const float  revoke = revoke_ ? outstanding(cur.score, keep) : 0.0f;
    // Default for a token no node on the fail chain continues: revoke.
    if (revoke != 0.0f) {
        for (int i = 0; i < n; ++i) {
            if (i != blank_id) {
                logits[i] -= revoke;
            }
        }
    }
    // Tokens that continue s itself.
    std::unordered_set<int32_t> seen;
    for (const auto & [t, child] : cur.next) {
        if (t >= 0 && t < n && t != blank_id) {
            logits[t] += revoke + nodes_[static_cast<size_t>(child)].arc;
            seen.insert(t);
        }
    }
    // Tokens that continue a shorter suffix match on the fail chain
    // (including fresh phrase starts at the root).
    if (s != 0) {
        int32_t f = cur.fail;
        while (true) {
            for (const auto & [t, child] : nodes_[static_cast<size_t>(f)].next) {
                if (t >= 0 && t < n && t != blank_id && seen.insert(t).second) {
                    // Without revocation, the suffix match's own
                    // accumulated score is not granted either: only its
                    // next arc.
                    logits[t] +=
                        revoke_ ? nodes_[static_cast<size_t>(child)].score : nodes_[static_cast<size_t>(child)].arc;
                }
            }
            if (f == 0) {
                break;
            }
            f = nodes_[static_cast<size_t>(f)].fail;
        }
    }
}

}  // namespace transcribe::parakeet
