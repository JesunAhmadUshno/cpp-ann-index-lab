// Unit tests for cpp-ann-index-lab. Hand-rolled harness, standard
// library only. Exit code 0 means every check passed.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "ann/brute_force.h"
#include "ann/dataset.h"
#include "ann/hnsw.h"
#include "ann/ivf.h"
#include "ann/vecmath.h"

static int failures = 0;
static int checks = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        ++checks;                                                       \
        if (!(cond)) {                                                  \
            ++failures;                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                               \
    } while (0)

namespace {

// Deterministic normalized gaussian dataset.
ann::Dataset make_data(std::size_t rows, std::size_t d, std::uint64_t seed) {
    ann::Dataset ds;
    ds.rows = rows;
    ds.dim = d;
    ds.data.resize(rows * d);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (std::size_t i = 0; i < rows; ++i) {
        float* v = &ds.data[i * d];
        float norm = 0.0f;
        for (std::size_t j = 0; j < d; ++j) { v[j] = gauss(rng); norm += v[j] * v[j]; }
        norm = std::sqrt(norm);
        for (std::size_t j = 0; j < d; ++j) v[j] /= norm;
    }
    return ds;
}

std::set<std::int32_t> as_set(const std::array<std::int32_t, 10>& a) {
    std::set<std::int32_t> s;
    for (auto x : a)
        if (x >= 0) s.insert(x);
    return s;
}

void test_vecmath() {
    const float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    CHECK(std::fabs(ann::inner_product(a, b, 4) - 70.0f) < 1e-5f);
    const float u[3] = {1.0f, 0.0f, 0.0f};
    const float v[3] = {0.0f, 1.0f, 0.0f};
    CHECK(std::fabs(ann::squared_l2(u, v, 3) - 2.0f) < 1e-6f);
    // Identity: squared L2 = 2 - 2 IP for unit vectors.
    const float w[3] = {0.6f, 0.8f, 0.0f};
    float lhs = ann::squared_l2(u, w, 3);
    float rhs = 2.0f - 2.0f * ann::inner_product(u, w, 3);
    CHECK(std::fabs(lhs - rhs) < 1e-6f);
}

void test_topk_ties() {
    ann::TopK<3> top;
    top.add(0.5f, 7);
    top.add(0.5f, 3);   // same sim, smaller id ranks first
    top.add(0.5f, 9);
    top.add(0.5f, 1);   // ties at the boundary: id 1 beats id 9
    top.add(0.4f, 0);   // below the cut
    CHECK(top.ids[0] == 1);
    CHECK(top.ids[1] == 3);
    CHECK(top.ids[2] == 7);
}

void test_brute_vs_gt() {
    auto ds = make_data(300, 24, 777);
    auto qs = make_data(25, 24, 778);
    for (std::size_t qi = 0; qi < qs.rows; ++qi) {
        auto a = ann::brute_force_search(ds.data.data(), ds.rows, ds.dim, qs.row(qi));
        auto b = ann::ground_truth_top10(ds.data.data(), ds.rows, ds.dim, qs.row(qi));
        CHECK(a == b);
    }
}

void test_ivf_full_probe_is_exact() {
    auto ds = make_data(400, 32, 1001);
    auto qs = make_data(20, 32, 1002);
    ann::IvfIndex idx(ds.data.data(), ds.rows, ds.dim, 8);
    idx.build();
    for (std::size_t qi = 0; qi < qs.rows; ++qi) {
        auto got = idx.search(qs.row(qi), 8);  // nprobe == nlist: full scan
        auto gt = ann::ground_truth_top10(ds.data.data(), ds.rows, ds.dim, qs.row(qi));
        CHECK(as_set(got) == as_set(gt));
    }
}

void test_hnsw_full_ef_is_exact() {
    auto ds = make_data(200, 32, 2001);
    auto qs = make_data(20, 32, 2002);
    ann::HnswIndex idx(ds.data.data(), ds.rows, ds.dim, 16);
    idx.build();
    for (std::size_t qi = 0; qi < qs.rows; ++qi) {
        // ef above N: the beam never prunes, so layer-0 search visits
        // the whole connected graph and must return the exact top-10.
        auto got = idx.search(qs.row(qi), 256);
        auto gt = ann::ground_truth_top10(ds.data.data(), ds.rows, ds.dim, qs.row(qi));
        CHECK(as_set(got) == as_set(gt));
    }
}

void test_kmeans_determinism() {
    auto ds = make_data(500, 16, 3001);
    ann::IvfIndex a(ds.data.data(), ds.rows, ds.dim, 8);
    ann::IvfIndex b(ds.data.data(), ds.rows, ds.dim, 8);
    a.build();
    b.build();
    CHECK(a.centroids() == b.centroids());
}

void test_loader_roundtrip() {
    auto ds = make_data(17, 8, 4001);
    const std::string path = "test_roundtrip_tmp.f32";
    ds.save_f32(path);
    auto back = ann::Dataset::load_f32(path, 17, 8);
    CHECK(back.rows == 17 && back.dim == 8);
    CHECK(back.data == ds.data);
    std::remove(path.c_str());
}

}  // namespace

int main() {
    test_vecmath();
    test_topk_ties();
    test_brute_vs_gt();
    test_ivf_full_probe_is_exact();
    test_hnsw_full_ef_is_exact();
    test_kmeans_determinism();
    test_loader_roundtrip();
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
