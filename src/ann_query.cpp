// ann_query: command-line front end for the ann library.
//
// Builds one index (IVF, graph, or exact brute force) over a raw
// float32 corpus file and answers top-10 inner-product queries from a
// raw float32 query file. Corpus and query files are headerless
// row-major float32, exactly the format written by
// tools/export_embeddings.py and tools/make_demo_data.py.
//
// The benchmark harness (ann_bench) stays the frozen study protocol;
// this tool is the everyday entry point for using the indexes.
//
// Usage:
//   ann_query --corpus corpus.f32 --queries queries.f32 \
//       --n 58999 --d 256 --nq 300 --index graph --m 32 --ef 64
//
// Options:
//   --corpus PATH   corpus vectors, n rows of d float32 (required)
//   --queries PATH  query vectors, nq rows of d float32 (required)
//   --n N           corpus row count (required)
//   --d D           vector dimension (required)
//   --nq NQ         query row count (required)
//   --index NAME    ivf | graph | brute (default graph)
//   --nlist N       IVF: number of inverted lists (default 512)
//   --nprobe N      IVF: lists probed per query (default 32)
//   --m M           graph: max links per node per layer (default 32)
//   --ef N          graph: search beam width (default 64)
//   --out PATH      also write the result lines to this file
//   --help          print this text
//
// Output: one line per query, "q<i>: id id ... id" with the top-10
// vector ids by inner product, best first. Lines starting with # are
// a run header (configuration and build time).

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ann/brute_force.h"
#include "ann/dataset.h"
#include "ann/hnsw.h"
#include "ann/ivf.h"

namespace {

std::string get_arg(int argc, char** argv, const std::string& name,
                    const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (name == argv[i]) return argv[i + 1];
    return fallback;
}

bool has_arg(int argc, char** argv, const std::string& name) {
    for (int i = 1; i < argc; ++i)
        if (name == argv[i]) return true;
    return false;
}

std::size_t get_count(int argc, char** argv, const std::string& name) {
    std::string v = get_arg(argc, argv, name, "");
    if (v.empty()) throw std::runtime_error("missing required " + name);
    return std::stoul(v);
}

void print_ids(std::ostream& os, std::size_t qi,
               const std::array<std::int32_t, 10>& ids) {
    os << "q" << qi << ":";
    for (std::int32_t id : ids) os << ' ' << id;
    os << '\n';
}

int run(int argc, char** argv) {
    if (has_arg(argc, argv, "--help")) {
        std::cout
            << "ann_query: build an ANN index over a float32 corpus and "
               "answer top-10 queries.\n"
            << "Required: --corpus PATH --queries PATH --n N --d D --nq NQ\n"
            << "Index: --index ivf|graph|brute (default graph)\n"
            << "IVF: --nlist N (512) --nprobe N (32); graph: --m N (32) "
               "--ef N (64)\n"
            << "Optional: --out PATH writes the result lines to a file.\n";
        return 0;
    }

    const std::string corpus_path = get_arg(argc, argv, "--corpus", "");
    const std::string queries_path = get_arg(argc, argv, "--queries", "");
    if (corpus_path.empty() || queries_path.empty())
        throw std::runtime_error("--corpus and --queries are required");
    const std::size_t n = get_count(argc, argv, "--n");
    const std::size_t d = get_count(argc, argv, "--d");
    const std::size_t nq = get_count(argc, argv, "--nq");
    const std::string index = get_arg(argc, argv, "--index", "graph");
    const std::string out_path = get_arg(argc, argv, "--out", "");

    ann::Dataset corpus = ann::Dataset::load_f32(corpus_path, n, d);
    ann::Dataset queries = ann::Dataset::load_f32(queries_path, nq, d);

    std::ofstream out_file;
    std::ostream* extra = nullptr;
    if (!out_path.empty()) {
        out_file.open(out_path);
        if (!out_file) throw std::runtime_error("cannot write " + out_path);
        extra = &out_file;
    }
    auto emit = [&](std::size_t qi, const std::array<std::int32_t, 10>& ids) {
        print_ids(std::cout, qi, ids);
        if (extra) print_ids(*extra, qi, ids);
    };

    const auto t0 = std::chrono::steady_clock::now();
    if (index == "ivf") {
        const std::size_t nlist =
            std::stoul(get_arg(argc, argv, "--nlist", "512"));
        const std::size_t nprobe =
            std::stoul(get_arg(argc, argv, "--nprobe", "32"));
        ann::IvfIndex idx(corpus.data.data(), n, d, nlist);
        idx.build();
        const auto t1 = std::chrono::steady_clock::now();
        std::cout << "# ann_query index=ivf nlist=" << nlist
                  << " nprobe=" << nprobe << " n=" << n << " d=" << d
                  << " build_s="
                  << std::chrono::duration<double>(t1 - t0).count() << "\n";
        for (std::size_t i = 0; i < nq; ++i)
            emit(i, idx.search(queries.row(i), nprobe));
    } else if (index == "graph") {
        const std::size_t m = std::stoul(get_arg(argc, argv, "--m", "32"));
        const std::size_t ef = std::stoul(get_arg(argc, argv, "--ef", "64"));
        ann::HnswIndex idx(corpus.data.data(), n, d, m);
        idx.build();
        const auto t1 = std::chrono::steady_clock::now();
        std::cout << "# ann_query index=graph M=" << m << " ef=" << ef
                  << " n=" << n << " d=" << d << " build_s="
                  << std::chrono::duration<double>(t1 - t0).count() << "\n";
        for (std::size_t i = 0; i < nq; ++i)
            emit(i, idx.search(queries.row(i), ef));
    } else if (index == "brute") {
        std::cout << "# ann_query index=brute (exact) n=" << n << " d=" << d
                  << "\n";
        for (std::size_t i = 0; i < nq; ++i)
            emit(i, ann::brute_force_search(corpus.data.data(), n, d,
                                            queries.row(i)));
    } else {
        throw std::runtime_error("unknown --index " + index +
                                 " (want ivf, graph or brute)");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ann_query: error: %s\n", e.what());
        return 1;
    }
}
