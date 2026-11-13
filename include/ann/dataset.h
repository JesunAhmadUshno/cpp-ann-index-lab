#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ann {

// Row-major float32 matrix loaded from a raw binary file (no header).
// The export script (tools/export_embeddings.py) writes exactly
// rows * dim float32 values, little-endian, matching the NumPy artifact.
struct Dataset {
    std::vector<float> data;
    std::size_t rows = 0;
    std::size_t dim = 0;

    const float* row(std::size_t i) const { return data.data() + i * dim; }

    static Dataset load_f32(const std::string& path, std::size_t rows_wanted,
                            std::size_t dim_) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open " + path);
        Dataset ds;
        ds.rows = rows_wanted;
        ds.dim = dim_;
        ds.data.resize(rows_wanted * dim_);
        std::size_t got = std::fread(ds.data.data(), sizeof(float),
                                     rows_wanted * dim_, f);
        std::fclose(f);
        if (got != rows_wanted * dim_)
            throw std::runtime_error("short read on " + path);
        return ds;
    }

    void save_f32(const std::string& path) const {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot write " + path);
        std::size_t put = std::fwrite(data.data(), sizeof(float), data.size(), f);
        std::fclose(f);
        if (put != data.size())
            throw std::runtime_error("short write on " + path);
    }
};

}  // namespace ann
