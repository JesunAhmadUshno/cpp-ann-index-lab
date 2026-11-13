#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace ann {

// Inner product of two float32 vectors of length d.
// Multiple accumulators keep the loop pipeline-friendly under -O3.
inline float inner_product(const float* a, const float* b, std::size_t d) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    std::size_t i = 0;
    for (; i + 4 <= d; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < d; ++i) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

// Squared L2 distance, used by the tests to cross-check the inner product
// on normalized vectors (||a-b||^2 = 2 - 2<a,b> when ||a|| = ||b|| = 1).
inline float squared_l2(const float* a, const float* b, std::size_t d) {
    float s = 0.0f;
    for (std::size_t i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        s += diff * diff;
    }
    return s;
}

// Streaming top-k under the total order (similarity desc, id asc).
// Inserting every candidate yields exactly the first k elements of a
// full sort under the same order, which is the ground-truth rule frozen
// in PREREGISTRATION.md (ties broken by the smaller row index).
template <std::size_t K>
struct TopK {
    std::array<float, K> sims{};
    std::array<std::int32_t, K> ids{};
    std::size_t count = 0;

    // True if (sim, id) ranks strictly inside the current top-k.
    bool admits(float sim, std::int32_t id) const {
        if (count < K) return true;
        float worst_sim = sims[K - 1];
        std::int32_t worst_id = ids[K - 1];
        return sim > worst_sim || (sim == worst_sim && id < worst_id);
    }

    void add(float sim, std::int32_t id) {
        if (!admits(sim, id)) return;
        std::size_t pos = count < K ? count++ : K - 1;
        // Shift down while the previous element ranks after (sim, id).
        while (pos > 0) {
            float ps = sims[pos - 1];
            std::int32_t pi = ids[pos - 1];
            bool prev_after = ps < sim || (ps == sim && pi > id);
            if (!prev_after) break;
            sims[pos] = ps;
            ids[pos] = pi;
            --pos;
        }
        sims[pos] = sim;
        ids[pos] = id;
    }
};

}  // namespace ann
