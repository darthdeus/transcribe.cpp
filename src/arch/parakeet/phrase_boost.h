// arch/parakeet/phrase_boost.h - phrase boosting (contextual biasing) for
// the transducer greedy decoders.
//
// Shallow fusion of a token-level Aho-Corasick "context graph" (icefall
// ContextGraph / NeMo GPU-PB style) into greedy RNN-T / TDT decoding.
// Each caller phrase (transcribe_run_params::vocabulary) is encoded with
// the model's SentencePiece vocab into a token-id path in a prefix trie.
// At every decode step a non-blank token t taken from trie state s scores
//
//     logit[t] + delta(s, t)
//
// where delta is the arc bonus when t extends the current partial match
// (weight * depth scale; for a token that breaks the match, the arc bonus
// of the longest suffix match it extends via the fail links, e.g. a fresh
// phrase start). The classic context-graph revocation (subtract the bonus
// accumulated by a partial match that breaks) is tracked but kept out of
// the greedy ranking by default; see PhraseBoost::build.
// Blank never moves the state and gets no bonus. Logits are raw joint
// outputs; adding to them is the same as adding in log-prob space since
// log_softmax is a uniform per-row shift.
//
// Opt-in: an empty phrase list builds no graph and the decoders take the
// unmodified argmax path (bit-identical output).

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace transcribe {
class Tokenizer;
}

namespace transcribe::parakeet {

class PhraseBoost {
  public:
    // Default per-token bonus in logit (nat) units; see docs in the
    // transcribe_parakeet_stream_ext::vocabulary_boost comment.
    static constexpr float kDefaultWeight = 3.0f;

    // Build from phrases. Each phrase is added as written and, when it
    // starts with a lowercase ASCII letter, with that letter capitalized
    // (sentence-initial form). Phrases that fail to encode, or that encode
    // to an unknown piece, are skipped. weight <= 0 builds an empty graph.
    void build(const transcribe::Tokenizer & tok, const std::vector<std::string> & phrases, float weight);

    bool empty() const { return nodes_.size() <= 1; }

    void clear() {
        nodes_.clear();
        weight_ = 0.0f;
    }

    // Score delta for taking token `t` from state `s`; writes the next
    // state to `*next`. Root is state 0.
    float delta(int32_t s, int32_t t, int32_t * next) const;

    // Add delta(s, t) in place to every non-blank logit in [0, n). Only
    // tokens with a non-zero delta are touched: arcs out of s and its fail
    // chain get their bonus; every other non-blank token gets the
    // revocation -score(s) (zero at the root).
    void apply(int32_t s, float * logits, int n, int blank_id) const;

    int n_phrases() const { return n_phrases_; }

  private:
    struct Node {
        std::unordered_map<int32_t, int32_t> next;
        int32_t                              fail  = 0;
        int32_t                              depth = 0;
        float                                arc   = 0.0f;  // bonus on the arc into this node
        float                                score = 0.0f;  // accumulated bonus from root
        bool                                 end   = false;
        // Nearest end node on the fail chain (or self): reaching it
        // commits the phrase bonus.
        int32_t                              out   = -1;
    };

    float   arc_bonus(int depth) const;
    int32_t goto_(int32_t s, int32_t t) const;

    std::vector<Node> nodes_;
    float             weight_    = 0.0f;
    float             first_     = 0.5f;   // first-token bonus scale
    bool              revoke_    = false;  // penalize breaking a partial match (see build())
    int               n_phrases_ = 0;
};

}  // namespace transcribe::parakeet
