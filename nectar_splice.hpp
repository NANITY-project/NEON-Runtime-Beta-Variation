#pragma once
// =============================================================================
// nectar_splice.hpp — Mid-stream injection support (NECTAR idle-loop doc
// §7, §8 build-order step 5).
//
// The doc frames mid-stream injection as the riskiest piece of the whole
// design ("no clean tick boundary anymore... the part most likely to be
// fragile"). That risk is real for an async/batched generation loop, but
// run_idle_loop() (NEON.cpp) is a plain synchronous token-at-a-time loop —
// which means there already IS a clean boundary to inject at: between
// finishing one token's forward pass and sampling the next. No pausing or
// interrupting in-flight work is actually required here; "splice" just
// means "forward some extra tokens through the model before you sample
// the next one," using the exact same ensure_cache_room()/forward
// machinery already used for regular generation.
//
// What IS still worth getting right, and what this header actually
// contains, is the bookkeeping: injected tokens become live cache
// positions exactly like generated ones, so they must be tracked in the
// same aging-buffer structure that feeds eviction-to-disk (§5.1) with the
// same one-entry-per-position invariant — get that wrong and either the
// disk store silently loses injected content, or the eviction accounting
// (discard count vs. aging.size()) drifts and the whole idle loop's memory
// bookkeeping becomes wrong in a way that's hard to notice until much
// later. That's the part this header exists to get right and keep
// testable independent of any model or live vision backend.
//
// Convention for injected content in the aging buffer: rather than trying
// to reuse the streaming per-token decoder (Tokenizer::decode_one, which
// has internal state for reconstructing partial UTF-8 sequences from
// *displayed* generation output and should not be fed silent, non-spoken
// injected tokens), an injected chunk of N tokens is pushed as N aging
// entries: the full annotation text on the FIRST entry, empty strings on
// the rest. This keeps the "one aging entry per live cache position"
// invariant exactly true (so eviction discard-count bookkeeping never
// drifts) without needing token-level granularity on text we already have
// as one known string — and it means the eviction-to-disk log reads the
// annotation at the position where it began, not scattered/duplicated.
// =============================================================================
#include <chrono>
#include <deque>
#include <string>

namespace mstream {

// Pushes an injected chunk into an aging buffer using the convention above.
// `aging` must be the same deque<string> used for regular per-token pushes
// (NEON.cpp's run_idle_loop) — after this call its size grows by exactly
// token_count, matching token_count new live cache positions having been
// added, which is what keeps ensure_cache_room()'s discard count valid
// against aging.size() regardless of whether the positions came from
// generation or injection.
inline void push_injected_text(std::deque<std::string>& aging,
                                size_t token_count,
                                const std::string& text) {
    if (token_count == 0) return;
    aging.push_back(text);
    for (size_t i = 1; i < token_count; ++i)
        aging.push_back("");
}

// Wall-clock cadence gate for polling an intermittent source (vision) from
// inside a tight per-token loop, without coupling the poll rate to
// generation speed (which varies with hardware/model size — a token-count
// based cadence would poll faster on a fast machine for no reason). Pure
// and clock-injectable so it's testable without sleeping in real tests.
class IntervalGate {
public:
    explicit IntervalGate(std::chrono::milliseconds interval)
        : interval_(interval), last_(std::chrono::steady_clock::time_point::min()) {}

    // True at most once per `interval`; calling this IS what starts the
    // next interval (so call it exactly once per loop iteration where you
    // intend to act on a true result — checking without consuming is not
    // supported, matching how the idle loop actually uses it).
    bool ready(std::chrono::steady_clock::time_point now) {
        if (last_ == std::chrono::steady_clock::time_point::min() || now - last_ >= interval_) {
            last_ = now;
            return true;
        }
        return false;
    }

private:
    std::chrono::milliseconds interval_;
    std::chrono::steady_clock::time_point last_;
};

} // namespace mstream
