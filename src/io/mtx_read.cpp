#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <numeric>
#include <vector>

#include "ichol/mtx_read.hpp"
#include "ichol/half.hpp"

namespace
{

    /**
     * Convert string to lowercase
     */
    inline std::string lower_copy(std::string s)
    {
        for (char &ch : s)
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return s;
    }

    /**
     * Check whether a line is blank or a comment line (starts with '%')
     */
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

    /**
     * Save lower tri + diag into COO format, first row-sorted, then column sorted.
     *
     * DO NOT check for definiteness, duplicatation, numerical value.
     * Check symmetry.
     * Expect at least the lower triangular is stored in mtx.
     */
    template <typename T>
    ichol::CooMatrix<T> mtx_to_coo(const std::string &path, bool verify)
    {
        std::ifstream in(path);
        if (!in)
            throw std::invalid_argument("ichol::io::mtx_to_coo: failed to open file: " + path);
        std::vector<char> io_buf(1u << 20);
        in.rdbuf()->pubsetbuf(io_buf.data(), static_cast<std::streamsize>(io_buf.size()));

        std::string line;
        int line_num = 0;
        if (!std::getline(in, line))
            throw std::invalid_argument("ichol::io::mtx_to_coo: empty file: " + path);
        ++line_num;

        Header hdr = parse_header_line(line);
        if (hdr.field == Field::Complex || hdr.field == Field::Pattern)
            throw std::invalid_argument("ichol::io::mtx_to_coo: real SPD reader supports only field {real, integer}");
        if (hdr.symmetry == Symmetry::SkewSymmetric)
            throw std::invalid_argument("ichol::io::mtx_to_coo: skew-symmetric not compatible with SPD requirement");
        if (hdr.symmetry == Symmetry::Hermitian)
            hdr.symmetry = Symmetry::Symmetric;

        while (std::getline(in, line))
        {
            ++line_num;
            if (!is_blank_or_comment(line))
                break;
        }
        if (!in)
            throw std::invalid_argument("ichol::io::mtx_to_coo: missing size line: " + path);

        int m = 0, n = 0;
        long long nnz_file = 0;
        {
            const char *p = line.c_str();
            const char *e = p + line.size();
            long long mm = parse_ll(p, e);
            long long nn = parse_ll(p, e);
            long long z = parse_ll(p, e);

            if (mm <= 0 || nn <= 0 || z < 0)
                throw std::invalid_argument("ichol::io::mtx_to_coo: invalid size line");
            if (mm > std::numeric_limits<int>::max() || nn > std::numeric_limits<int>::max())
                throw std::invalid_argument("ichol::io::mtx_to_coo: dimensions exceed int");
            m = static_cast<int>(mm);
            n = static_cast<int>(nn);
            nnz_file = z;
        }

        if (m != n)
            throw std::invalid_argument("ichol::io::mtx_to_coo: expected square matrix for lower-tri+diag storage");

        std::size_t reserve_nnz = static_cast<std::size_t>(nnz_file);
        if (hdr.symmetry == Symmetry::General)
            reserve_nnz = static_cast<std::size_t>(nnz_file / 2 + n);

        ichol::CooMatrix<T> out;
        out.num_cols = n;
        out.num_rows = n;
        out.col_ind.reserve(reserve_nnz);
        out.row_ind.reserve(reserve_nnz);
        out.values.reserve(reserve_nnz);

        // Make sure no diag is missing
        std::vector<unsigned char> diag_seen(static_cast<std::size_t>(n), 0);

        auto keep_and_map = [&](int &r, int &c) -> bool
        {
            if (hdr.symmetry == Symmetry::General)
            {
                if (r < c)
                    return false; // keep only lower+diag
                return true;
            }
            else
            {
                // symmetric: map to lower (robust if file stores upper)
                if (r < c)
                    std::swap(r, c);
            }

            return true;
        };

        // Read entries
        while (std::getline(in, line))
        {
            ++line_num;
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
                throw std::invalid_argument("ichol::io::mtx_to_coo: parse error at line " + std::to_string(line_num) + ": " + ex.what());
            }

            if (rr_ll <= 0 || cc_ll <= 0)
                throw std::invalid_argument("ichol::io::mtx_to_coo: indices must be 1-based positive (line " + std::to_string(line_num) + ")");

            if (rr_ll > n || cc_ll > n)
            {
                if (verify)
                    throw std::invalid_argument("ichol::io::mtx_to_coo: index out of range at line " + std::to_string(line_num));
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
                throw std::invalid_argument("ichol::io::mtx_to_coo: value parse error at line " + std::to_string(line_num) + ": " + ex.what());
            }

            if (!keep_and_map(r, c))
                continue;

            if (r == c)
                diag_seen[static_cast<std::size_t>(r)] = 1;

            out.row_ind.push_back(r);
            out.col_ind.push_back(c);
            out.values.push_back(v);
        }

        // Check if missing diag entry
        for (int i = 0; i < n; ++i)
        {
            if (!diag_seen[static_cast<std::size_t>(i)])
                throw std::invalid_argument("ichol::io::mtx_to_coo: missing diagonal entry at i=i=" + std::to_string(i + 1) + " (1-based)");
        }

        if (out.row_ind.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("ichol::io::mtx_to_coo: nnz exceeds int capacity");

        // after rows/cols/vals are filled
        std::vector<int> perm(out.row_ind.size());
        std::iota(perm.begin(), perm.end(), 0);

        std::sort(perm.begin(), perm.end(), [&](int a, int b)
                  {
                        if (out.row_ind[a] != out.row_ind[b]) {
                            return out.row_ind[a] < out.row_ind[b];
                        }
    return out.col_ind[a] < out.col_ind[b]; });

        auto apply_perm = [&](auto &v)
        {
            using V = typename std::decay_t<decltype(v)>::value_type;
            std::vector<V> tmp;
            tmp.reserve(v.size());
            for (int k : perm)
                tmp.push_back(v[k]);
            v.swap(tmp);
        };

        apply_perm(out.row_ind);
        apply_perm(out.col_ind);
        apply_perm(out.values);

        out.nnz = static_cast<int>(out.values.size());
        return out;
    }

    template <typename T>
    ichol::CsrMatrix<T> coo_to_csr(ichol::CooMatrix<T> coo_in)
    {
        ichol::CsrMatrix<T> csr_out;
        csr_out.num_cols = coo_in.num_cols;
        csr_out.num_rows = coo_in.num_rows;
        csr_out.nnz = coo_in.nnz;

        csr_out.row_ptr.assign(static_cast<std::size_t>(csr_out.num_rows) + 1, 0);

        // count nnz per row
        for (int r : coo_in.row_ind)
            ++csr_out.row_ptr[static_cast<std::size_t>(r) + 1];

        // exclusive prefix sum
        for (std::size_t i = 0; i + 1 < csr_out.row_ptr.size(); ++i)
            csr_out.row_ptr[i + 1] += csr_out.row_ptr[i];

        csr_out.col_ind = std::move(coo_in.col_ind);
        csr_out.values = std::move(coo_in.values);

        return csr_out;
    }

    template <typename T>
    ichol::CscMatrix<T> coo_to_csc(ichol::CooMatrix<T> coo_in)
    {
        ichol::CscMatrix<T> csc_out;
        csc_out.num_cols = coo_in.num_cols;
        csc_out.num_rows = coo_in.num_rows;
        csc_out.nnz = coo_in.nnz;

        const std::size_t nnz = static_cast<std::size_t>(csc_out.nnz);

        // build col_ptr
        csc_out.col_ptr.assign(static_cast<std::size_t>(csc_out.num_cols) + 1, 0);
        for (int c : coo_in.col_ind)
            ++csc_out.col_ptr[static_cast<std::size_t>(c) + 1];

        for (std::size_t j = 0; j + 1 < csc_out.col_ptr.size(); ++j)
            csc_out.col_ptr[j + 1] += csc_out.col_ptr[j];

        // scatter
        csc_out.row_ind.resize(nnz);
        csc_out.values.resize(nnz);

        auto next = csc_out.col_ptr; // same type as col_ptr

        for (std::size_t k = 0; k < nnz; ++k)
        {
            const int r = coo_in.row_ind[k];
            const int c = coo_in.col_ind[k];
            const auto pos = next[static_cast<std::size_t>(c)]++;

            csc_out.row_ind[static_cast<std::size_t>(pos)] = r;
            csc_out.values[static_cast<std::size_t>(pos)] = coo_in.values[k];
        }

        // sort rows within each column (and permute values)
        for (int cj = 0; cj < csc_out.num_cols; ++cj)
        {
            const std::size_t j = static_cast<std::size_t>(cj);
            const std::size_t b = static_cast<std::size_t>(csc_out.col_ptr[j]);
            const std::size_t e = static_cast<std::size_t>(csc_out.col_ptr[j + 1]);
            const std::size_t len = e - b;

            if (len <= 1)
                continue;

            std::vector<std::size_t> perm(len);
            std::iota(perm.begin(), perm.end(), 0);

            std::sort(perm.begin(), perm.end(),
                      [&](std::size_t a, std::size_t d)
                      {
                          return csc_out.row_ind[b + a] < csc_out.row_ind[b + d];
                      });

            std::vector<int> rtmp;
            std::vector<T> vtmp;
            rtmp.reserve(len);
            vtmp.reserve(len);

            for (std::size_t t : perm)
            {
                rtmp.push_back(csc_out.row_ind[b + t]);
                vtmp.push_back(csc_out.values[b + t]);
            }

            for (std::size_t t = 0; t < len; ++t)
            {
                csc_out.row_ind[b + t] = rtmp[t];
                csc_out.values[b + t] = vtmp[t];
            }
        }

        return csc_out;
    }
}

namespace ichol
{
    namespace io
    {
        template <typename T>
        CsrMatrix<T> mtx_to_csr(const std::string &path, bool verify)
        {
            auto coo = mtx_to_coo<T>(path, verify);
            return coo_to_csr(std::move(coo));
        }

        template <typename T>
        CscMatrix<T> mtx_to_csc(const std::string &path, bool verify)
        {
            auto coo = mtx_to_coo<T>(path, verify);
            return coo_to_csc(std::move(coo));
        }

        template CsrMatrix<double> mtx_to_csr<double>(const std::string &path, bool verify);
        template CsrMatrix<float> mtx_to_csr<float>(const std::string &path, bool verify);
        template CsrMatrix<half_float::half> mtx_to_csr<half_float::half>(const std::string &path, bool verify);

        template CscMatrix<double> mtx_to_csc<double>(const std::string &path, bool verify);
        template CscMatrix<float> mtx_to_csc<float>(const std::string &path, bool verify);
        template CscMatrix<half_float::half> mtx_to_csc<half_float::half>(const std::string &path, bool verify);
    }

} // namespace ichol