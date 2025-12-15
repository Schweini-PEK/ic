#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "mtx_read.hpp"

namespace mtx_detail
{

    inline std::string lower_copy(std::string s)
    {
        for (char &ch : s)
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return s;
    }

    inline bool is_blank_or_comment(const std::string &line)
    {
        for (char ch : line)
        {
            if (!std::isspace(static_cast<unsigned char>(ch)))
                return ch == '%';
        }
        return true;
    }

    enum class Field
    {
        Real,
        Integer,
        Pattern,
        Complex
    };
    enum class Symmetry
    {
        General,
        Symmetric,
        SkewSymmetric,
        Hermitian
    };

    struct Header
    {
        Field field = Field::Real;
        Symmetry symmetry = Symmetry::General;
    };

    [[noreturn]] inline void abort_missing_diag(const std::string &path, int n, int first_missing_1based)
    {
        std::fprintf(stderr,
                     "ERROR: MTX missing diagonal entry at i=i=%d (1-based). "
                     "Aborting. File: %s (n=%d)\n",
                     first_missing_1based, path.c_str(), n);
        std::abort();
    }

    inline Header parse_header_line(const std::string &line)
    {
        // %%MatrixMarket matrix coordinate {real|integer|complex|pattern} {general|symmetric|skew-symmetric|hermitian}
        std::istringstream iss(line);
        std::string banner, object, format, field, sym;
        iss >> banner >> object >> format >> field >> sym;

        banner = lower_copy(banner);
        object = lower_copy(object);
        format = lower_copy(format);
        field = lower_copy(field);
        sym = lower_copy(sym);

        if (banner != "%%matrixmarket")
            throw std::runtime_error("MTX: invalid banner (expected %%MatrixMarket)");
        if (object != "matrix")
            throw std::runtime_error("MTX: only 'matrix' object supported");
        if (format != "coordinate")
            throw std::runtime_error("MTX: only 'coordinate' format supported");

        Header h;

        if (field == "real")
            h.field = Field::Real;
        else if (field == "integer")
            h.field = Field::Integer;
        else if (field == "pattern")
            h.field = Field::Pattern;
        else if (field == "complex")
            h.field = Field::Complex;
        else
            throw std::runtime_error("MTX: unknown field type: " + field);

        if (sym == "general")
            h.symmetry = Symmetry::General;
        else if (sym == "symmetric")
            h.symmetry = Symmetry::Symmetric;
        else if (sym == "skew-symmetric")
            h.symmetry = Symmetry::SkewSymmetric;
        else if (sym == "hermitian")
            h.symmetry = Symmetry::Hermitian;
        else
            throw std::runtime_error("MTX: unknown symmetry type: " + sym);

        return h;
    }

    inline long long parse_ll(const char *&p, const char *end)
    {
        while (p < end && std::isspace(static_cast<unsigned char>(*p)))
            ++p;
        char *q = nullptr;
        long long v = std::strtoll(p, &q, 10);
        if (q == p)
            throw std::runtime_error("MTX: failed to parse integer");
        p = q;
        return v;
    }

    inline long double parse_ld(const char *&p, const char *end)
    {
        while (p < end && std::isspace(static_cast<unsigned char>(*p)))
            ++p;
        char *q = nullptr;
        long double v = std::strtold(p, &q);
        if (q == p)
            throw std::runtime_error("MTX: failed to parse floating-point");
        p = q;
        return v;
    }

    template <typename T>
    struct Entry
    {
        int col;
        T val;
    };

} // namespace mtx_detail

namespace ichol
{

    template <typename T>
    CSR<T> readMTXtoCSR(const std::string &path, bool verify)
    {
        static_assert(std::is_floating_point_v<T>,
                      "readMTXtoCSR<T>: real SPD reader expects T=float/double.");

        using namespace mtx_detail;

        std::ifstream in(path);
        if (!in)
            throw std::runtime_error("MTX: failed to open file: " + path);

        // Larger IO buffer for speed on big files.
        std::vector<char> io_buf(1u << 20);
        in.rdbuf()->pubsetbuf(io_buf.data(), static_cast<std::streamsize>(io_buf.size()));

        std::string line;
        int line_no = 0;

        if (!std::getline(in, line))
            throw std::runtime_error("MTX: empty file: " + path);
        ++line_no;

        Header hdr = parse_header_line(line);

        // Real SPD only: accept real/integer; accept general/symmetric; treat hermitian as symmetric if real.
        if (hdr.field == Field::Complex || hdr.field == Field::Pattern)
            throw std::runtime_error("MTX: real SPD reader supports only field {real, integer}");
        if (hdr.symmetry == Symmetry::SkewSymmetric)
            throw std::runtime_error("MTX: skew-symmetric not compatible with SPD requirement");
        if (hdr.symmetry == Symmetry::Hermitian)
            hdr.symmetry = Symmetry::Symmetric;

        // Skip comments to size line
        while (std::getline(in, line))
        {
            ++line_no;
            if (!is_blank_or_comment(line))
                break;
        }
        if (!in)
            throw std::runtime_error("MTX: missing size line: " + path);

        // Parse size line: M N NNZ
        int M = 0, N = 0;
        long long nnz_file = 0;
        {
            const char *p = line.c_str();
            const char *e = p + line.size();
            long long m = parse_ll(p, e);
            long long n = parse_ll(p, e);
            long long z = parse_ll(p, e);

            if (m <= 0 || n <= 0 || z < 0)
                throw std::runtime_error("MTX: invalid size line");
            if (m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max())
                throw std::runtime_error("MTX: dimensions exceed int");
            M = static_cast<int>(m);
            N = static_cast<int>(n);
            nnz_file = z;
        }

        if (M != N)
            throw std::runtime_error("MTX: expected square matrix for lower-tri+diag storage");

        // Reserve: for general, about half; for symmetric, keep almost all.
        std::size_t reserve_nnz = static_cast<std::size_t>(nnz_file);
        if (hdr.symmetry == Symmetry::General)
            reserve_nnz = static_cast<std::size_t>(nnz_file / 2 + N);

        std::vector<int> rows;
        std::vector<int> cols;
        std::vector<T> vals;
        rows.reserve(reserve_nnz);
        cols.reserve(reserve_nnz);
        vals.reserve(reserve_nnz);

        std::vector<unsigned char> diag_seen(static_cast<std::size_t>(N), 0);

        auto keep_and_map = [&](int &r, int &c) -> bool
        {
            if (hdr.symmetry == Symmetry::General)
            {
                if (r < c)
                    return false; // keep only lower+diag
                return true;
            }
            // symmetric: map to lower (robust if file stores upper)
            if (r < c)
                std::swap(r, c);
            return true;
        };

        // Read entries
        while (std::getline(in, line))
        {
            ++line_no;
            if (is_blank_or_comment(line))
                continue;

            const char *p = line.c_str();
            const char *e = p + line.size();

            long long rr_ll, cc_ll;
            try
            {
                rr_ll = parse_ll(p, e);
                cc_ll = parse_ll(p, e);
            }
            catch (const std::exception &ex)
            {
                throw std::runtime_error("MTX: parse error at line " + std::to_string(line_no) + ": " + ex.what());
            }

            if (rr_ll <= 0 || cc_ll <= 0)
                throw std::runtime_error("MTX: indices must be 1-based positive (line " + std::to_string(line_no) + ")");

            if (rr_ll > M || cc_ll > N)
            {
                if (verify)
                    throw std::runtime_error("MTX: index out of range at line " + std::to_string(line_no));
                continue;
            }

            int r = static_cast<int>(rr_ll - 1);
            int c = static_cast<int>(cc_ll - 1);

            T v{};
            try
            {
                if (hdr.field == Field::Integer)
                {
                    long long iv = parse_ll(p, e);
                    v = static_cast<T>(iv);
                }
                else
                { // real
                    long double dv = parse_ld(p, e);
                    v = static_cast<T>(dv);
                }
            }
            catch (const std::exception &ex)
            {
                throw std::runtime_error("MTX: value parse error at line " + std::to_string(line_no) + ": " + ex.what());
            }

            if (!keep_and_map(r, c))
                continue;

            if (r == c)
                diag_seen[static_cast<std::size_t>(r)] = 1;

            rows.push_back(r);
            cols.push_back(c);
            vals.push_back(v);
        }

        // Abort if any diagonal is missing.
        for (int i = 0; i < N; ++i)
        {
            if (!diag_seen[static_cast<std::size_t>(i)])
                abort_missing_diag(path, N, i + 1);
        }

        if (rows.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("MTX: nnz exceeds int capacity");

        // Build CSR row_ptr
        CSR<T> out;
        out.num_rows = N;
        out.num_cols = N;
        out.nnz = static_cast<int>(rows.size());
        out.row_ptr.assign(static_cast<std::size_t>(N + 1), 0);

        for (int r : rows)
            ++out.row_ptr[static_cast<std::size_t>(r + 1)];
        for (int i = 0; i < N; ++i)
            out.row_ptr[static_cast<std::size_t>(i + 1)] += out.row_ptr[static_cast<std::size_t>(i)];

        if (!verify)
        {
            // Fast path: no sorting/merging.
            out.col_ind.assign(static_cast<std::size_t>(out.nnz), 0);
            out.values.assign(static_cast<std::size_t>(out.nnz), T{});
            std::vector<int> next = out.row_ptr;

            for (std::size_t k = 0; k < rows.size(); ++k)
            {
                int r = rows[k];
                int pos = next[static_cast<std::size_t>(r)]++;
                out.col_ind[static_cast<std::size_t>(pos)] = cols[k];
                out.values[static_cast<std::size_t>(pos)] = vals[k];
            }
            return out;
        }

        // Verify path: scatter to entries, sort each row, merge duplicates, rebuild CSR.
        std::vector<Entry<T>> entries(static_cast<std::size_t>(out.nnz));
        {
            std::vector<int> next = out.row_ptr;
            for (std::size_t k = 0; k < rows.size(); ++k)
            {
                int r = rows[k];
                int pos = next[static_cast<std::size_t>(r)]++;
                entries[static_cast<std::size_t>(pos)] = Entry<T>{cols[k], vals[k]};
            }
        }

        for (int r = 0; r < N; ++r)
        {
            int b = out.row_ptr[static_cast<std::size_t>(r)];
            int e = out.row_ptr[static_cast<std::size_t>(r + 1)];
            std::sort(entries.begin() + b, entries.begin() + e,
                      [](const Entry<T> &a, const Entry<T> &b)
                      { return a.col < b.col; });
        }

        std::vector<int> new_row_ptr(static_cast<std::size_t>(N + 1), 0);
        std::vector<Entry<T>> new_entries;
        new_entries.reserve(entries.size());

        new_row_ptr[0] = 0;
        for (int r = 0; r < N; ++r)
        {
            int b = out.row_ptr[static_cast<std::size_t>(r)];
            int e = out.row_ptr[static_cast<std::size_t>(r + 1)];

            for (int i = b; i < e;)
            {
                int col = entries[static_cast<std::size_t>(i)].col;
                T sum = entries[static_cast<std::size_t>(i)].val;
                int j = i + 1;
                while (j < e && entries[static_cast<std::size_t>(j)].col == col)
                {
                    sum += entries[static_cast<std::size_t>(j)].val;
                    ++j;
                }
                new_entries.push_back(Entry<T>{col, sum});
                i = j;
            }
            new_row_ptr[static_cast<std::size_t>(r + 1)] = static_cast<int>(new_entries.size());
        }

        out.row_ptr = std::move(new_row_ptr);
        out.nnz = static_cast<int>(new_entries.size());

        out.col_ind.resize(static_cast<std::size_t>(out.nnz));
        out.values.resize(static_cast<std::size_t>(out.nnz));
        for (int k = 0; k < out.nnz; ++k)
        {
            out.col_ind[static_cast<std::size_t>(k)] = new_entries[static_cast<std::size_t>(k)].col;
            out.values[static_cast<std::size_t>(k)] = new_entries[static_cast<std::size_t>(k)].val;
        }

        return out;
    }

    template CSR<double> readMTXtoCSR<double>(const std::string &path, bool verify);
    template CSR<float> readMTXtoCSR<float>(const std::string &path, bool verify);

} // namespace ichol
