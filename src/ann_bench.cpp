// Benchmark harness for cpp-ann-index-lab.
//
// Protocol: PREREGISTRATION.md. One untimed warm-up pass per index
// configuration, then 5 timed repetitions per search cell; per-query
// latency by std::chrono::steady_clock; nearest-rank percentiles per
// repetition; recall@10 against brute-force ground truth.
//
// Modes:
//   real:      --corpus F32 --queries F32 --n N --d D --nq Q
//              [--gt-out PATH] [--out PATH]
//   synthetic: --synthetic (small deterministic slice for CI smoke)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "ann/brute_force.h"
#include "ann/dataset.h"
#include "ann/hnsw.h"
#include "ann/ivf.h"

namespace {

using Clock = std::chrono::steady_clock;

double to_ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

// Nearest-rank percentile over one repetition's latencies (ascending).
double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    std::size_t rank = static_cast<std::size_t>(std::ceil(p / 100.0 * v.size()));
    if (rank == 0) rank = 1;
    if (rank > v.size()) rank = v.size();
    return v[rank - 1];
}

struct Stats {
    double mean = 0.0, stddev = 0.0;
};

Stats mean_std(const std::vector<double>& xs) {
    Stats s;
    double sum = std::accumulate(xs.begin(), xs.end(), 0.0);
    s.mean = sum / xs.size();
    double sq = 0.0;
    for (double x : xs) sq += (x - s.mean) * (x - s.mean);
    s.stddev = xs.size() > 1 ? std::sqrt(sq / (xs.size() - 1)) : 0.0;
    return s;
}

double median(std::vector<double> xs) {
    std::sort(xs.begin(), xs.end());
    std::size_t m = xs.size();
    return m % 2 ? xs[m / 2] : 0.5 * (xs[m / 2 - 1] + xs[m / 2]);
}

struct CellResult {
    std::string index;
    std::string params;          // JSON fragment with the config
    double build_s = 0.0;        // median build time
    int builds = 0;
    std::size_t structure_bytes = 0;
    std::vector<double> recall, p50, p99;  // per repetition
};

std::string fmt(double x, int prec = 6) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", prec, x);
    return buf;
}

// Runs one search callable over the query set with the frozen protocol.
template <typename SearchFn>
void run_cell(SearchFn&& search, const ann::Dataset& queries,
              const std::vector<std::array<std::int32_t, 10>>& gt, int reps,
              CellResult& cell) {
    auto run_once = [&](bool timed, double& recall_out, double& p50_out,
                        double& p99_out) {
        std::vector<double> lat;
        lat.reserve(queries.rows);
        double recall_sum = 0.0;
        for (std::size_t qi = 0; qi < queries.rows; ++qi) {
            const float* q = queries.row(qi);
            auto t0 = Clock::now();
            auto got = search(q);
            auto t1 = Clock::now();
            if (timed) lat.push_back(to_ms(t1 - t0));
            int hit = 0;
            for (int a : got) {
                if (a < 0) continue;
                for (int b : gt[qi])
                    if (a == b) { ++hit; break; }
            }
            recall_sum += hit / 10.0;
        }
        recall_out = recall_sum / queries.rows;
        if (timed) {
            p50_out = percentile(lat, 50.0);
            p99_out = percentile(lat, 99.0);
        }
    };
    double r, a, b;
    run_once(false, r, a, b);  // warm-up pass, untimed
    for (int rep = 0; rep < reps; ++rep) {
        double rec, p50, p99;
        run_once(true, rec, p50, p99);
        cell.recall.push_back(rec);
        cell.p50.push_back(p50);
        cell.p99.push_back(p99);
    }
}

void write_json(const std::string& path, const std::string& body) {
    std::ofstream out(path);
    out << body;
}

std::string cell_json(const CellResult& c, std::size_t n, std::size_t d) {
    Stats rs = mean_std(c.recall), s50 = mean_std(c.p50), s99 = mean_std(c.p99);
    double total_per_vec =
        static_cast<double>(c.structure_bytes) / n + d * sizeof(float);
    std::string s = "    {\"index\": \"" + c.index + "\", \"params\": " + c.params +
        ", \"build_time_s_median\": " + fmt(c.build_s) +
        ", \"builds\": " + std::to_string(c.builds) +
        ", \"structure_bytes\": " + std::to_string(c.structure_bytes) +
        ", \"total_bytes_per_vector\": " + fmt(total_per_vec, 2) +
        ", \"recall_mean\": " + fmt(rs.mean) +
        ", \"recall_std\": " + fmt(rs.stddev) +
        ", \"p50_ms_mean\": " + fmt(s50.mean) +
        ", \"p50_ms_std\": " + fmt(s50.stddev) +
        ", \"p99_ms_mean\": " + fmt(s99.mean) +
        ", \"p99_ms_std\": " + fmt(s99.stddev) +
        ", \"reps\": [";
    for (std::size_t i = 0; i < c.recall.size(); ++i) {
        if (i) s += ", ";
        s += "{\"recall\": " + fmt(c.recall[i]) + ", \"p50_ms\": " + fmt(c.p50[i]) +
             ", \"p99_ms\": " + fmt(c.p99[i]) + "}";
    }
    s += "]}\n";
    return s;
}

std::string get_arg(int argc, char** argv, const std::string& name,
                    const std::string& dflt = "") {
    for (int i = 1; i + 1 < argc; ++i)
        if (name == argv[i]) return argv[i + 1];
    return dflt;
}

bool has_arg(int argc, char** argv, const std::string& name) {
    for (int i = 1; i < argc; ++i)
        if (name == argv[i]) return true;
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    ann::Dataset corpus, queries;
    bool synthetic = has_arg(argc, argv, "--synthetic");
    int reps = 5;
    std::string out_path = get_arg(argc, argv, "--out", "");
    std::string gt_path = get_arg(argc, argv, "--gt-out", "");

    if (synthetic) {
        // Deterministic small slice for CI smoke: gaussian vectors,
        // L2-normalized, fixed seed. Not a study result.
        const std::size_t n = 2000, d = 64, nq = 50;
        std::mt19937_64 rng(424242ULL);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        auto fill = [&](ann::Dataset& ds, std::size_t rows) {
            ds.rows = rows;
            ds.dim = d;
            ds.data.resize(rows * d);
            for (std::size_t i = 0; i < rows; ++i) {
                float* v = &ds.data[i * d];
                float norm = 0.0f;
                for (std::size_t j = 0; j < d; ++j) { v[j] = gauss(rng); norm += v[j] * v[j]; }
                norm = std::sqrt(norm);
                for (std::size_t j = 0; j < d; ++j) v[j] /= norm;
            }
        };
        fill(corpus, n);
        fill(queries, nq);
        reps = 2;
    } else {
        std::size_t n = std::stoul(get_arg(argc, argv, "--n", "58999"));
        std::size_t d = std::stoul(get_arg(argc, argv, "--d", "256"));
        std::size_t nq = std::stoul(get_arg(argc, argv, "--nq", "300"));
        corpus = ann::Dataset::load_f32(get_arg(argc, argv, "--corpus"), n, d);
        queries = ann::Dataset::load_f32(get_arg(argc, argv, "--queries"), nq, d);
    }

    const std::size_t n = corpus.rows, d = corpus.dim;

    // Ground truth by the full-sort implementation.
    std::vector<std::array<std::int32_t, 10>> gt(queries.rows);
    for (std::size_t qi = 0; qi < queries.rows; ++qi)
        gt[qi] = ann::ground_truth_top10(corpus.data.data(), n, d, queries.row(qi));

    if (!gt_path.empty()) {
        std::ofstream out(gt_path);
        out << "[";
        for (std::size_t qi = 0; qi < queries.rows; ++qi) {
            if (qi) out << ",";
            out << "[";
            for (int i = 0; i < 10; ++i) {
                if (i) out << ",";
                out << gt[qi][i];
            }
            out << "]";
        }
        out << "]\n";
    }

    std::vector<CellResult> cells;

    // Brute-force baseline cell.
    {
        CellResult c;
        c.index = "brute_force";
        c.params = "{}";
        c.builds = 1;
        c.build_s = 0.0;
        c.structure_bytes = 0;
        run_cell([&](const float* q) {
            return ann::brute_force_search(corpus.data.data(), n, d, q);
        }, queries, gt, reps, c);
        cells.push_back(c);
    }

    // IVF cells.
    std::vector<std::size_t> nlists = synthetic ? std::vector<std::size_t>{8}
                                                : std::vector<std::size_t>{128, 512};
    std::vector<std::size_t> nprobes = synthetic
        ? std::vector<std::size_t>{1, 2, 4, 8}
        : std::vector<std::size_t>{1, 2, 4, 8, 16, 32, 64};
    for (std::size_t nlist : nlists) {
        ann::IvfIndex index(corpus.data.data(), n, d, nlist);
        std::vector<double> builds;
        int n_builds = synthetic ? 1 : 5;
        for (int b = 0; b < n_builds; ++b) {
            ann::IvfIndex fresh(corpus.data.data(), n, d, nlist);
            auto t0 = Clock::now();
            fresh.build();
            builds.push_back(to_ms(Clock::now() - t0) / 1000.0);
            if (b == n_builds - 1) index = std::move(fresh);
        }
        for (std::size_t nprobe : nprobes) {
            if (nprobe > nlist) continue;
            CellResult c;
            c.index = "ivf";
            c.params = "{\"nlist\": " + std::to_string(nlist) +
                       ", \"nprobe\": " + std::to_string(nprobe) + "}";
            c.build_s = median(builds);
            c.builds = n_builds;
            c.structure_bytes = index.structure_bytes();
            run_cell([&](const float* q) { return index.search(q, nprobe); },
                     queries, gt, reps, c);
            cells.push_back(c);
        }
    }

    // Graph (HNSW-style) cells.
    std::vector<std::size_t> ms = synthetic ? std::vector<std::size_t>{8}
                                            : std::vector<std::size_t>{16, 32};
    std::vector<std::size_t> efs = synthetic ? std::vector<std::size_t>{16, 64}
                                             : std::vector<std::size_t>{16, 32, 64, 128, 256};
    for (std::size_t m : ms) {
        std::vector<double> builds;
        int n_builds = synthetic ? 1 : 3;
        // Build via a holder so the timed final build is the measured one.
        auto holder = std::make_unique<ann::HnswIndex>(corpus.data.data(), n, d, m);
        for (int b = 0; b < n_builds; ++b) {
            auto fresh = std::make_unique<ann::HnswIndex>(corpus.data.data(), n, d, m);
            auto t0 = Clock::now();
            fresh->build();
            builds.push_back(to_ms(Clock::now() - t0) / 1000.0);
            if (b == n_builds - 1) holder = std::move(fresh);
        }
        for (std::size_t ef : efs) {
            CellResult c;
            c.index = "hnsw";
            c.params = "{\"M\": " + std::to_string(m) +
                       ", \"ef\": " + std::to_string(ef) +
                       ", \"efConstruction\": 200}";
            c.build_s = median(builds);
            c.builds = n_builds;
            c.structure_bytes = holder->structure_bytes();
            run_cell([&](const float* q) { return holder->search(q, ef); },
                     queries, gt, reps, c);
            cells.push_back(c);
        }
    }

    std::string json = "{\n  \"env\": {\"mode\": \"" +
        std::string(synthetic ? "synthetic" : "real") +
        "\", \"n\": " + std::to_string(n) +
        ", \"dim\": " + std::to_string(d) +
        ", \"n_queries\": " + std::to_string(queries.rows) +
        ", \"reps\": " + std::to_string(reps) + "},\n  \"cells\": [\n";
    for (std::size_t i = 0; i < cells.size(); ++i) {
        json += cell_json(cells[i], n, d);
        if (i + 1 < cells.size()) json += ",";
    }
    json += "  ]\n}\n";

    if (!out_path.empty()) write_json(out_path, json);
    std::printf("%s", json.c_str());
    return 0;
}
