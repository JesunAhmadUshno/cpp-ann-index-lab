#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "ann/vecmath.h"

namespace ann {

// Exact brute-force top-10 by inner product: the baseline arm and the
// source of latency numbers for the speed ratios in H1.
inline std::array<std::int32_t, 10> brute_force_search(
    const float* data, std::size_t n, std::size_t d, const float* q) {
    TopK<10> top;
    for (std::size_t i = 0; i < n; ++i) {
        float s = inner_product(q, data + i * d, d);
        top.add(s, static_cast<std::int32_t>(i));
    }
    return top.ids;
}

// Independent ground-truth implementation: full sort of all similarities
// under (similarity desc, id asc). Used to produce the ground-truth file
// and cross-checked against brute_force_search in the test suite, so the
// recall numbers rest on two agreeing implementations plus the NumPy
// validation in tools/validate_ground_truth.py.
inline std::array<std::int32_t, 10> ground_truth_top10(
    const float* data, std::size_t n, std::size_t d, const float* q) {
    std::vector<float> sims(n);
    for (std::size_t i = 0; i < n; ++i)
        sims[i] = inner_product(q, data + i * d, d);
    std::vector<std::int32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](std::int32_t a, std::int32_t b) {
        if (sims[a] != sims[b]) return sims[a] > sims[b];
        return a < b;
    });
    std::array<std::int32_t, 10> out{};
    for (int i = 0; i < 10; ++i) out[i] = idx[i];
    return out;
}

}  // namespace ann
