#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "ann/vecmath.h"

namespace ann {

// Inverted file index. Training is spherical k-means: assignment by
// maximum inner product against L2-renormalized centroids, matching the
// normalized corpus geometry (cosine as inner product). Protocol frozen
// in PREREGISTRATION.md: 20 Lloyd iterations, training sample = first
// min(N, 10000) corpus vectors, seed 20261113, initial centroids are
// distinct sampled training vectors.
class IvfIndex {
public:
    static constexpr std::uint64_t kTrainSeed = 20261113ULL;
    static constexpr int kTrainIters = 20;
    static constexpr std::size_t kTrainSample = 10000;

    IvfIndex(const float* data, std::size_t n, std::size_t d, std::size_t nlist)
        : data_(data), n_(n), d_(d), nlist_(nlist) {}

    void build() {
        train();
        assign_all();
    }

    // Copy of the trained centroids, exposed for the determinism test.
    const std::vector<float>& centroids() const { return centroids_; }

    std::array<std::int32_t, 10> search(const float* q, std::size_t nprobe) const {
        // Score every centroid, take the nprobe best (ties: smaller
        // centroid index first, via a full sort of a small array).
        std::vector<std::pair<float, std::int32_t>> scored;
        scored.reserve(nlist_);
        for (std::size_t c = 0; c < nlist_; ++c)
            scored.emplace_back(inner_product(q, &centroids_[c * d_], d_),
                                static_cast<std::int32_t>(c));
        std::partial_sort(scored.begin(), scored.begin() + nprobe, scored.end(),
                          [](const auto& a, const auto& b) {
                              if (a.first != b.first) return a.first > b.first;
                              return a.second < b.second;
                          });
        TopK<10> top;
        for (std::size_t p = 0; p < nprobe; ++p) {
            std::int32_t c = scored[p].second;
            const std::size_t begin = static_cast<std::size_t>(offsets_[c]);
            const std::size_t end = static_cast<std::size_t>(offsets_[c + 1]);
            for (std::size_t j = begin; j < end; ++j) {
                std::int32_t id = ids_[j];
                float s = inner_product(q, data_ + static_cast<std::size_t>(id) * d_, d_);
                top.add(s, id);
            }
        }
        return top.ids;
    }

    // Structure bytes owned by the index (excludes the corpus vector
    // array, which RESULTS.md attributes separately at N x dim x 4).
    std::size_t structure_bytes() const {
        std::size_t bytes = 0;
        bytes += centroids_.capacity() * sizeof(float);
        bytes += ids_.capacity() * sizeof(std::int32_t);
        bytes += offsets_.capacity() * sizeof(std::int32_t);
        bytes += nlist_ * sizeof(float);  // centroid score scratch
        return bytes;
    }

private:
    void train() {
        const std::size_t m = std::min(n_, kTrainSample);
        std::vector<std::int32_t> perm(m);
        std::iota(perm.begin(), perm.end(), 0);
        std::mt19937_64 rng(kTrainSeed);
        std::shuffle(perm.begin(), perm.end(), rng);
        centroids_.assign(nlist_ * d_, 0.0f);
        for (std::size_t c = 0; c < nlist_; ++c) {
            const float* src = data_ + static_cast<std::size_t>(perm[c]) * d_;
            std::copy(src, src + d_, &centroids_[c * d_]);
        }
        std::vector<float> sums(nlist_ * d_);
        std::vector<std::int32_t> assign(m);
        for (int iter = 0; iter < kTrainIters; ++iter) {
            for (std::size_t i = 0; i < m; ++i) {
                const float* v = data_ + i * d_;
                float best = -std::numeric_limits<float>::infinity();
                std::int32_t bc = 0;
                for (std::size_t c = 0; c < nlist_; ++c) {
                    float s = inner_product(v, &centroids_[c * d_], d_);
                    if (s > best) { best = s; bc = static_cast<std::int32_t>(c); }
                }
                assign[i] = bc;
            }
            std::fill(sums.begin(), sums.end(), 0.0f);
            for (std::size_t i = 0; i < m; ++i) {
                const float* v = data_ + i * d_;
                float* dst = &sums[static_cast<std::size_t>(assign[i]) * d_];
                for (std::size_t j = 0; j < d_; ++j) dst[j] += v[j];
            }
            for (std::size_t c = 0; c < nlist_; ++c) {
                float* cen = &centroids_[c * d_];
                const float* sum = &sums[c * d_];
                float norm = 0.0f;
                for (std::size_t j = 0; j < d_; ++j) norm += sum[j] * sum[j];
                norm = std::sqrt(norm);
                if (norm > 0.0f)
                    for (std::size_t j = 0; j < d_; ++j) cen[j] = sum[j] / norm;
                // Empty cluster: keep the previous centroid (deterministic).
            }
        }
    }

    void assign_all() {
        std::vector<std::vector<std::int32_t>> lists(nlist_);
        for (std::size_t i = 0; i < n_; ++i) {
            const float* v = data_ + i * d_;
            float best = -std::numeric_limits<float>::infinity();
            std::int32_t bc = 0;
            for (std::size_t c = 0; c < nlist_; ++c) {
                float s = inner_product(v, &centroids_[c * d_], d_);
                if (s > best) { best = s; bc = static_cast<std::int32_t>(c); }
            }
            lists[bc].push_back(static_cast<std::int32_t>(i));
        }
        offsets_.assign(nlist_ + 1, 0);
        for (std::size_t c = 0; c < nlist_; ++c)
            offsets_[c + 1] = offsets_[c] + static_cast<std::int32_t>(lists[c].size());
        ids_.reserve(n_);
        for (std::size_t c = 0; c < nlist_; ++c)
            ids_.insert(ids_.end(), lists[c].begin(), lists[c].end());
    }

    const float* data_;
    std::size_t n_;
    std::size_t d_;
    std::size_t nlist_;
    std::vector<float> centroids_;
    std::vector<std::int32_t> ids_;
    std::vector<std::int32_t> offsets_;  // nlist_ + 1 entries
};

}  // namespace ann
