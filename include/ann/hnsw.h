#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include "ann/vecmath.h"

namespace ann {

// Graph index in the HNSW style (Malkov and Yashunin), implemented from
// scratch here. Similarity is inner product on L2-normalized vectors, so
// "nearest" means largest inner product throughout. Neighbour selection
// is the simple variant: keep the M closest candidates (labelled as such
// in METHODS notes in RESULTS.md; the diversity heuristic is not used).
// Protocol frozen in PREREGISTRATION.md: efConstruction = 200, level
// multiplier 1 / ln(M), seed 20261114, layer-0 link cap 2M, upper-layer
// link cap M, insertion in corpus order 0..N-1.
class HnswIndex {
public:
    static constexpr std::uint64_t kLevelSeed = 20261114ULL;
    static constexpr std::size_t kEfConstruction = 200;

    HnswIndex(const float* data, std::size_t n, std::size_t d, std::size_t m)
        : data_(data), n_(n), d_(d), m_(m),
          ml_(1.0 / std::log(static_cast<double>(m))),
          rng_(kLevelSeed), uni_(0.0, 1.0),
          levels_(n, 0), links_(n), visited_(n, 0) {}

    void build() {
        for (std::size_t i = 0; i < n_; ++i) insert(static_cast<std::int32_t>(i));
        finalize();
    }

    std::array<std::int32_t, 10> search(const float* q, std::size_t ef) const {
        std::int32_t ep = entry_;
        float ep_sim = inner_product(q, vec(ep), d_);
        for (int level = max_level_; level > 0; --level) {
            auto w = search_layer(q, ep, ep_sim, 1, level);
            ep = w[0].second;
            ep_sim = w[0].first;
        }
        auto w = search_layer(q, ep, ep_sim, ef, 0);
        // w is sorted by (sim desc, id asc) by search_layer's contract.
        std::array<std::int32_t, 10> out{};
        std::size_t take = std::min<std::size_t>(w.size(), 10);
        for (std::size_t i = 0; i < take; ++i) out[i] = w[i].second;
        // If fewer than 10 were found (tiny graphs), pad with -1.
        for (std::size_t i = take; i < 10; ++i) out[i] = -1;
        return out;
    }

    std::size_t node_level(std::int32_t id) const { return static_cast<std::size_t>(levels_[id]); }

    // Structure bytes owned by the index (excludes the corpus vector
    // array, attributed separately in RESULTS.md). Counted after
    // finalize() shrunk every link vector to its exact size.
    std::size_t structure_bytes() const {
        std::size_t bytes = 0;
        bytes += levels_.capacity() * sizeof(std::int32_t);
        bytes += visited_.capacity() * sizeof(std::int32_t);
        for (const auto& node : links_) {
            bytes += sizeof(std::vector<std::int32_t>) * node.size();
            for (const auto& layer : node)
                bytes += layer.capacity() * sizeof(std::int32_t);
        }
        bytes += links_.capacity() * sizeof(std::vector<std::vector<std::int32_t>>);
        return bytes;
    }

private:
    const float* vec(std::int32_t id) const {
        return data_ + static_cast<std::size_t>(id) * d_;
    }

    int random_level() {
        double u = uni_(rng_);
        if (u <= 0.0) u = std::numeric_limits<double>::min();
        int level = static_cast<int>(-std::log(u) * ml_);
        return level;
    }

    std::size_t cap_for(int level) const { return level == 0 ? 2 * m_ : m_; }

    using SimId = std::pair<float, std::int32_t>;

    // Greedy / beam search inside one layer. Returns the ef best found,
    // sorted by (sim desc, id asc). Candidates and results are bounded
    // by ef. Standard HNSW SEARCH-LAYER with similarities inverted.
    std::vector<SimId> search_layer(const float* q, std::int32_t ep,
                                    float ep_sim, std::size_t ef,
                                    int level) const {
        ++visit_stamp_;
        // Candidate queue: best (largest sim) on top.
        auto cand_cmp = [](const SimId& a, const SimId& b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second > b.second;
        };
        // Result set: worst (smallest sim) on top.
        auto res_cmp = [](const SimId& a, const SimId& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        };
        std::priority_queue<SimId, std::vector<SimId>, decltype(cand_cmp)> cand(cand_cmp);
        std::priority_queue<SimId, std::vector<SimId>, decltype(res_cmp)> res(res_cmp);
        visited_[ep] = visit_stamp_;
        cand.emplace(ep_sim, ep);
        res.emplace(ep_sim, ep);
        while (!cand.empty()) {
            SimId c = cand.top();
            cand.pop();
            if (res.size() >= ef && c.first < res.top().first) break;
            if (level > levels_[c.second]) continue;  // node absent here
            for (std::int32_t v : links_[c.second][level]) {
                if (visited_[v] == visit_stamp_) continue;
                visited_[v] = visit_stamp_;
                float s = inner_product(q, vec(v), d_);
                if (res.size() < ef || s > res.top().first ||
                    (s == res.top().first && v < res.top().second)) {
                    cand.emplace(s, v);
                    res.emplace(s, v);
                    if (res.size() > ef) res.pop();
                }
            }
        }
        std::vector<SimId> out;
        out.reserve(res.size());
        while (!res.empty()) { out.push_back(res.top()); res.pop(); }
        std::sort(out.begin(), out.end(), [](const SimId& a, const SimId& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });
        return out;
    }

    // Simple neighbour selection: the cap closest candidates.
    std::vector<std::int32_t> select_nearest(std::vector<SimId> cands,
                                             std::size_t cap) const {
        std::sort(cands.begin(), cands.end(),
                  [](const SimId& a, const SimId& b) {
                      if (a.first != b.first) return a.first > b.first;
                      return a.second < b.second;
                  });
        if (cands.size() > cap) cands.resize(cap);
        std::vector<std::int32_t> out;
        out.reserve(cands.size());
        for (const auto& c : cands) out.push_back(c.second);
        return out;
    }

    void insert(std::int32_t u) {
        int lu = random_level();
        levels_[u] = lu;
        links_[u].assign(static_cast<std::size_t>(lu) + 1, {});
        if (entry_ < 0) {
            entry_ = u;
            max_level_ = lu;
            return;
        }
        const float* q = vec(u);
        std::int32_t ep = entry_;
        float ep_sim = inner_product(q, vec(ep), d_);
        for (int level = max_level_; level > lu; --level) {
            auto w = search_layer(q, ep, ep_sim, 1, level);
            ep = w[0].second;
            ep_sim = w[0].first;
        }
        int top = std::min(lu, max_level_);
        for (int level = top; level >= 0; --level) {
            auto w = search_layer(q, ep, ep_sim, kEfConstruction, level);
            // Exclude u itself (not yet linked, but be safe) and pick.
            std::vector<SimId> cands;
            for (const auto& p : w)
                if (p.second != u) cands.push_back(p);
            std::vector<std::int32_t> neighbours =
                select_nearest(cands, cap_for(level));
            links_[u][level] = neighbours;
            for (std::int32_t v : neighbours) {
                auto& vlinks = links_[v][level];
                vlinks.push_back(u);
                if (vlinks.size() > cap_for(level)) {
                    std::vector<SimId> vc;
                    vc.reserve(vlinks.size());
                    for (std::int32_t x : vlinks)
                        vc.emplace_back(inner_product(vec(v), vec(x), d_), x);
                    links_[v][level] = select_nearest(vc, cap_for(level));
                }
            }
            if (!w.empty()) { ep = w[0].second; ep_sim = w[0].first; }
        }
        if (lu > max_level_) {
            entry_ = u;
            max_level_ = lu;
        }
    }

    void finalize() {
        for (auto& node : links_)
            for (auto& layer : node) layer.shrink_to_fit();
    }

    const float* data_;
    std::size_t n_;
    std::size_t d_;
    std::size_t m_;
    double ml_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uni_;
    std::vector<std::int32_t> levels_;
    std::vector<std::vector<std::vector<std::int32_t>>> links_;
    std::int32_t entry_ = -1;
    int max_level_ = 0;
    mutable std::vector<std::int32_t> visited_;
    mutable std::int32_t visit_stamp_ = 0;
};

}  // namespace ann
