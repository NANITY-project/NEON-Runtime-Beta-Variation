#pragma once
// =============================================================================
// rawllm_util.hpp  —  NUMA-aware ThreadPool
// =============================================================================
#include "rawllm_common.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace util {

inline void bind_thread_numa(int node) {
#ifdef USE_NUMA
    if (numa_available() < 0) return;
    int max_node = numa_max_node();
    node = node % (max_node + 1);
    struct bitmask* m = numa_bitmask_alloc(max_node + 1);
    numa_bitmask_setbit(m, node);
    numa_set_membind(m);
    numa_run_on_node(node);
    numa_bitmask_free(m);
#else
    (void)node;
#endif
}

// Flush denormals to zero / treat denormal inputs as zero on the calling
// thread. Softmax tails (exp of very negative scores) and tiny activations
// otherwise hit microcode-assist denormal arithmetic (100+ cycles/op); seen
// making the attention loop ~3x slower. Harmless for inference accuracy.
inline void set_ftz_daz() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_setcsr(_mm_getcsr() | 0x8040u);   // FTZ (bit 15) | DAZ (bit 6)
#endif
}

inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
}

// Low-latency pool: workers spin briefly on an atomic pending-task counter
// before parking on a condition variable, and wait() runs queued tasks on the
// calling thread instead of sleeping. A decode step does ~5 submit/wait
// barriers per layer; with a pure mutex+condvar pool every barrier paid a
// futex wake-up (tens of microseconds) per worker, which is a large fraction
// of a ~25 ms token on a 0.5B model.
//
// API is unchanged: submit(f) from the owning thread, wait() to join.
class ThreadPool {
public:
    explicit ThreadPool(int n_workers, bool numa_bind = false) {
        n_workers = std::max(1, n_workers);
        set_ftz_daz();                       // the thread that calls wait() runs tasks too
        workers_.reserve(n_workers);
        for (int i = 0; i < n_workers; ++i) {
            workers_.emplace_back([this, i, numa_bind] {
                set_ftz_daz();
                if (numa_bind) bind_thread_numa(i);
                worker_loop();
            });
        }
    }

    ~ThreadPool() {
        wait();
        { std::lock_guard<std::mutex> lk(mu_); stop_.store(true, std::memory_order_seq_cst); }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }

    template<typename F>
    void submit(F&& f) {
        {
            SpinGuard g(qlock_);
            q_.emplace_back(std::forward<F>(f));
        }
        outstanding_.fetch_add(1, std::memory_order_seq_cst);
        pending_.fetch_add(1, std::memory_order_seq_cst);   // published AFTER the push
        // Lost-wakeup safety: a worker increments sleepers_ and then re-checks
        // pending_ under mu_; we read sleepers_ after bumping pending_ (all
        // seq_cst), so either we see the sleeper and notify under mu_, or it
        // sees our pending_ bump before it blocks.
        if (sleepers_.load(std::memory_order_seq_cst) > 0) {
            { std::lock_guard<std::mutex> lk(mu_); }
            cv_.notify_one();
        }
    }

    // Blocks until every submitted task has finished. Helps by running queued
    // tasks on the calling thread.
    void wait() {
        unsigned spins = 0;
        while (outstanding_.load(std::memory_order_acquire) > 0) {
            if (try_run_one()) { spins = 0; continue; }
            if (++spins < 4000) cpu_relax();
            else std::this_thread::yield();
        }
    }

    int size() const { return static_cast<int>(workers_.size()); }

private:
    struct SpinGuard {
        std::atomic_flag& f;
        explicit SpinGuard(std::atomic_flag& fl) : f(fl) {
            unsigned n = 0;
            while (f.test_and_set(std::memory_order_acquire)) {
                if (++n < 256) cpu_relax(); else std::this_thread::yield();
            }
        }
        ~SpinGuard() { f.clear(std::memory_order_release); }
    };

    bool try_run_one() {
        int p = pending_.load(std::memory_order_acquire);
        while (p > 0) {
            if (pending_.compare_exchange_weak(p, p - 1, std::memory_order_acq_rel)) {
                std::function<void()> task;
                {
                    SpinGuard g(qlock_);
                    task = std::move(q_.front());
                    q_.pop_front();
                }
                try { task(); } catch (...) { std::terminate(); }   // same fate as the old worker threads
                outstanding_.fetch_sub(1, std::memory_order_acq_rel);
                return true;
            }
        }
        return false;
    }

    void worker_loop() {
        using clk = std::chrono::steady_clock;
        constexpr auto kSpin = std::chrono::microseconds(100);
        for (;;) {
            if (try_run_one()) continue;
            if (stop_.load(std::memory_order_acquire)) return;
            // Spin phase: catches the next barrier of a decode step without a futex round trip.
            auto t0 = clk::now();
            bool got = false;
            for (unsigned i = 0;; ++i) {
                if (pending_.load(std::memory_order_acquire) > 0) { got = true; break; }
                if (stop_.load(std::memory_order_relaxed)) return;
                cpu_relax();
                if ((i & 31) == 31 && clk::now() - t0 > kSpin) break;
            }
            if (got) continue;
            // Park phase.
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] {
                    return stop_.load(std::memory_order_seq_cst) ||
                           pending_.load(std::memory_order_seq_cst) > 0;
                });
            }
            sleepers_.fetch_sub(1, std::memory_order_seq_cst);
        }
    }

    std::vector<std::thread>             workers_;
    std::deque<std::function<void()>>    q_;
    std::atomic_flag                     qlock_ = ATOMIC_FLAG_INIT;
    std::mutex                           mu_;
    std::condition_variable              cv_;
    std::atomic<int>                     pending_{0};      // queued, not yet claimed
    std::atomic<int>                     outstanding_{0};  // queued + running
    std::atomic<int>                     sleepers_{0};
    std::atomic<bool>                    stop_{false};
};

} // namespace util
