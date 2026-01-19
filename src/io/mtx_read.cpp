#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ichol/mtx_read.hpp"
#include "ichol/half.hpp"

namespace
{
    // Lowercase helper for MatrixMarket tokens.
    inline std::string lower_copy(std::string s)
    {
        for (char &ch : s)
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return s;
    }

    // Skip blank lines and '%' comments.
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

    // Parse the MatrixMarket header and map tokens to enums.
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

    // Parse an integer from a line buffer (advances pointer).
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

    // Parse a floating-point from a line buffer (advances pointer).
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

    struct SizeLine
    {
        int n = 0;
        long long nnz_file = 0;
    };

    // Scan until the first non-blank, non-comment line.
    inline std::string read_first_noncomment_line(std::ifstream &in, int &line_num)
    {
        std::string line;
        while (std::getline(in, line))
        {
            ++line_num;
            if (!is_blank_or_comment(line))
                return line;
        }
        return {};
    }

    // Parse the "rows cols nnz" size line and validate square dimensions.
    inline SizeLine parse_size_line(const std::string &line, const std::string &path, int line_num)
    {
        const char *p = line.c_str();
        const char *e = p + line.size();

        long long mm = parse_ll(p, e);
        long long nn = parse_ll(p, e);
        long long z = parse_ll(p, e);

        if (mm <= 0 || nn <= 0 || z < 0)
            throw std::invalid_argument("ichol::io::mtx_to_coo: invalid size line at " + path + ":" + std::to_string(line_num));
        if (mm != nn)
            throw std::invalid_argument("ichol::io::mtx_to_coo: expected square matrix: " + path);

        if (mm > std::numeric_limits<int>::max())
            throw std::invalid_argument("ichol::io::mtx_to_coo: dimensions exceed int: " + path);

        SizeLine s;
        s.n = static_cast<int>(mm);
        s.nnz_file = z;
        return s;
    }

    // Keep only the lower triangle (and map symmetric entries into lower).
    inline bool map_keep_lower(int &r, int &c, Symmetry sym)
    {
        if (sym == Symmetry::General)
        {
            if (r < c)
                return false;
            return true;
        }
        // symmetric / hermitian: map to lower
        if (r < c)
            std::swap(r, c);
        return true;
    }

    template <typename T>
    // Stable row-major ordering for CSR conversion and deterministic output.
    inline void sort_coo_by_row_then_col(ichol::matrix::CooMatrix<T> &coo)
    {
        const std::size_t nnz = coo.row_ind.size();
        std::vector<std::size_t> perm(nnz);
        std::iota(perm.begin(), perm.end(), 0);

        std::sort(perm.begin(), perm.end(),
                  [&](std::size_t a, std::size_t b)
                  {
                      if (coo.row_ind[a] != coo.row_ind[b])
                          return coo.row_ind[a] < coo.row_ind[b];
                      return coo.col_ind[a] < coo.col_ind[b];
                  });

        auto apply_perm = [&](auto &v)
        {
            using V = typename std::decay_t<decltype(v)>::value_type;
            std::vector<V> tmp;
            tmp.reserve(v.size());
            for (std::size_t k : perm)
                tmp.push_back(v[k]);
            v.swap(tmp);
        };

        apply_perm(coo.row_ind);
        apply_perm(coo.col_ind);
        apply_perm(coo.values);
    }

    template <typename T>
    void read_numeric_lower_coo(std::ifstream &in,
                                const std::string &path,
                                const Header &hdr,
                                int n,
                                bool verify,
                                ichol::matrix::CooMatrix<T> &out)
    {
        // Numeric MTX path: keep lower triangle and require explicit diagonal.
        // Require diagonal to exist in-file (old behavior)
        std::vector<unsigned char> diag_seen(static_cast<std::size_t>(n), 0);

        std::string line;
        int line_num = 0; // caller already consumed header/size; this is local-only for error context

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
                throw std::invalid_argument("ichol::io::mtx_to_coo: parse error in " + path + ": " + ex.what());
            }

            if (rr_ll <= 0 || cc_ll <= 0)
                throw std::invalid_argument("ichol::io::mtx_to_coo: indices must be 1-based positive: " + path);

            if (rr_ll > n || cc_ll > n)
            {
                if (verify)
                    throw std::invalid_argument("ichol::io::mtx_to_coo: index out of range: " + path);
                continue;
            }

            int r = static_cast<int>(rr_ll - 1);
            int c = static_cast<int>(cc_ll - 1);

            // Drop upper-tri entries and map symmetric cases into lower.
            if (!map_keep_lower(r, c, hdr.symmetry))
                continue;

            T v{};
            try
            {
                if (hdr.field == Field::Integer)
                {
                    long long iv = parse_ll(p, e);
                    v = static_cast<T>(iv);
                }
                else
                {
                    long double dv = parse_ld(p, e);
                    v = static_cast<T>(dv);
                }
            }
            catch (const std::exception &ex)
            {
                throw std::invalid_argument("ichol::io::mtx_to_coo: value parse error in " + path + ": " + ex.what());
            }

            if (r == c)
                diag_seen[static_cast<std::size_t>(r)] = 1;

            out.row_ind.push_back(r);
            out.col_ind.push_back(c);
            out.values.push_back(v);
        }

        for (int i = 0; i < n; ++i)
        {
            if (!diag_seen[static_cast<std::size_t>(i)])
                throw std::invalid_argument("ichol::io::mtx_to_coo: missing diagonal entry at i=" + std::to_string(i + 1) + " (1-based) in " + path);
        }
    }

    struct PatternGraph
    {
        std::vector<std::uint64_t> edges_lower; // packed (row<<32)|col with row>col
        std::vector<int> degree;                // undirected degree
    };

    inline std::uint64_t pack_edge_u32(int r, int c)
    {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(r)) << 32) |
               static_cast<std::uint64_t>(static_cast<std::uint32_t>(c));
    }

    inline int unpack_row(std::uint64_t key) { return static_cast<int>(static_cast<std::uint32_t>(key >> 32)); }
    inline int unpack_col(std::uint64_t key) { return static_cast<int>(static_cast<std::uint32_t>(key & 0xffffffffu)); }

    inline PatternGraph read_pattern_graph(std::ifstream &in,
                                           const std::string &path,
                                           const Header &hdr,
                                           int n,
                                           bool verify,
                                           long long nnz_file)
    {
        // Pattern MTX path: treat entries as an unweighted undirected graph.
        PatternGraph g;
        g.degree.assign(static_cast<std::size_t>(n), 0);
        g.edges_lower.reserve(static_cast<std::size_t>(nnz_file));

        std::string line;
        int line_num = 0;

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
                throw std::invalid_argument("ichol::io::mtx_to_coo: parse error in " + path + ": " + ex.what());
            }

            if (rr_ll <= 0 || cc_ll <= 0)
                throw std::invalid_argument("ichol::io::mtx_to_coo: indices must be 1-based positive: " + path);

            if (rr_ll > n || cc_ll > n)
            {
                if (verify)
                    throw std::invalid_argument("ichol::io::mtx_to_coo: index out of range: " + path);
                continue;
            }

            int r = static_cast<int>(rr_ll - 1);
            int c = static_cast<int>(cc_ll - 1);

            if (!map_keep_lower(r, c, hdr.symmetry))
                continue;

            if (r == c)
            {
                // Self-loops cancel in L = D - A (diag adds w and subtracts w), ignore.
                continue;
            }

            // r>c guaranteed after map_keep_lower except for General case where r<c is dropped
            if (r < c)
                continue;

            // Treat as undirected adjacency edge with weight 1
            ++g.degree[static_cast<std::size_t>(r)];
            ++g.degree[static_cast<std::size_t>(c)];

            g.edges_lower.push_back(pack_edge_u32(r, c));
        }

        return g;
    }

    template <typename T>
    ichol::matrix::CooMatrix<T> build_shifted_laplacian_lower_coo(PatternGraph &&g,
                                                                  int n,
                                                                  double alpha)
    {
        // Collapse duplicates and build lower-tri COO for L = (D - A) + alpha I.
        std::sort(g.edges_lower.begin(), g.edges_lower.end());

        ichol::matrix::CooMatrix<T> out;
        out.num_rows = n;
        out.num_cols = n;

        // Worst case: no duplicates -> nnz = |E_lower| + n(diag)
        out.row_ind.reserve(g.edges_lower.size() + static_cast<std::size_t>(n));
        out.col_ind.reserve(g.edges_lower.size() + static_cast<std::size_t>(n));
        out.values.reserve(g.edges_lower.size() + static_cast<std::size_t>(n));

        std::size_t idx = 0;
        const std::size_t m = g.edges_lower.size();

        for (int r = 0; r < n; ++r)
        {
            // Emit all off-diagonals in this row, compressed with multiplicity.
            while (idx < m && unpack_row(g.edges_lower[idx]) == r)
            {
                const std::uint64_t key = g.edges_lower[idx];
                const int c = unpack_col(key);

                int count = 1;
                ++idx;
                while (idx < m && g.edges_lower[idx] == key)
                {
                    ++count;
                    ++idx;
                }

                // Off-diagonal value = -A_rc. For unweighted adjacency with multiplicity, A_rc = count.
                out.row_ind.push_back(r);
                out.col_ind.push_back(c);
                out.values.push_back(static_cast<T>(-static_cast<long double>(count)));
            }

            // Diagonal: degree + alpha
            const long double diag = static_cast<long double>(g.degree[static_cast<std::size_t>(r)]) + static_cast<long double>(alpha);
            out.row_ind.push_back(r);
            out.col_ind.push_back(r);
            out.values.push_back(static_cast<T>(diag));
        }

        if (out.values.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("ichol::io::mtx_to_coo: nnz exceeds int capacity");
        out.nnz = static_cast<int>(out.values.size());
        return out;
    }

    /**
     * Read MTX and return lower-tri + diag COO.
     *
     * - Numeric (real/integer): keep old behavior (require diagonal in file).
     * - Pattern: interpret as adjacency and build shifted Laplacian L = (D - A) + alpha I.
     */
    template <typename T>
    ichol::matrix::CooMatrix<T> mtx_to_coo(const std::string &path, bool verify, double alpha)
    {
        std::ifstream in(path);
        if (!in)
            throw std::invalid_argument("ichol::io::mtx_to_coo: failed to open file: " + path);

        // Use a larger IO buffer to reduce small-read overhead.
        std::vector<char> io_buf(1u << 20);
        in.rdbuf()->pubsetbuf(io_buf.data(), static_cast<std::streamsize>(io_buf.size()));

        std::string line;
        int line_num = 0;

        if (!std::getline(in, line))
            throw std::invalid_argument("ichol::io::mtx_to_coo: empty file: " + path);
        ++line_num;

        Header hdr = parse_header_line(line);

        if (hdr.symmetry == Symmetry::SkewSymmetric)
            throw std::invalid_argument("ichol::io::mtx_to_coo: skew-symmetric not compatible with SPD requirement: " + path);
        if (hdr.symmetry == Symmetry::Hermitian)
            hdr.symmetry = Symmetry::Symmetric;

        // Find size line
        std::string size_line = read_first_noncomment_line(in, line_num);
        if (!in || size_line.empty())
            throw std::invalid_argument("ichol::io::mtx_to_coo: missing size line: " + path);

        SizeLine sz = parse_size_line(size_line, path, line_num);
        const int n = sz.n;

        if (hdr.field == Field::Complex)
            throw std::invalid_argument("ichol::io::mtx_to_coo: complex field not supported: " + path);

        if (hdr.field == Field::Pattern)
        {
            // Build SPD operator from graph: L = (D - A) + alpha I.
            // alpha should be > 0 to make it strictly SPD even if the graph is disconnected.
            PatternGraph g = read_pattern_graph(in, path, hdr, n, verify, sz.nnz_file);
            auto coo = build_shifted_laplacian_lower_coo<T>(std::move(g), n, alpha);
            // Already generated row-major with col-sorted per row; no extra sort required.
            return coo;
        }

        // Numeric path (old behavior)
        if (hdr.field != Field::Real && hdr.field != Field::Integer)
            throw std::invalid_argument("ichol::io::mtx_to_coo: unsupported field type: " + path);

        ichol::matrix::CooMatrix<T> out;
        out.num_rows = n;
        out.num_cols = n;

        // Reserve heuristic: general -> keep about half; symmetric -> keep as-is
        std::size_t reserve_nnz = static_cast<std::size_t>(sz.nnz_file);
        if (hdr.symmetry == Symmetry::General)
            reserve_nnz = static_cast<std::size_t>(sz.nnz_file / 2 + n);

        out.row_ind.reserve(reserve_nnz);
        out.col_ind.reserve(reserve_nnz);
        out.values.reserve(reserve_nnz);

        read_numeric_lower_coo(in, path, hdr, n, verify, out);

        // Ensure row-major order before CSR conversion.
        sort_coo_by_row_then_col(out);

        if (out.row_ind.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("ichol::io::mtx_to_coo: nnz exceeds int capacity");
        out.nnz = static_cast<int>(out.values.size());
        return out;
    }

    template <typename T>
    ichol::matrix::CsrMatrix<T> coo_to_csr(const ichol::matrix::CooMatrix<T> &coo_in)
    {
        // Classic COO->CSR: count rows, prefix-sum, then scatter entries.
        ichol::matrix::CsrMatrix<T> csr;
        csr.num_rows = coo_in.num_rows;
        csr.num_cols = coo_in.num_cols;
        csr.nnz = coo_in.nnz;

        const std::size_t n = static_cast<std::size_t>(csr.num_rows);
        const std::size_t nnz = static_cast<std::size_t>(csr.nnz);

        csr.row_ptr.assign(n + 1, 0);

        for (std::size_t k = 0; k < nnz; ++k)
            ++csr.row_ptr[static_cast<std::size_t>(coo_in.row_ind[k]) + 1];

        for (std::size_t i = 0; i < n; ++i)
            csr.row_ptr[i + 1] += csr.row_ptr[i];

        csr.col_ind.resize(nnz);
        csr.values.resize(nnz);

        auto next = csr.row_ptr;
        for (std::size_t k = 0; k < nnz; ++k)
        {
            const int r = coo_in.row_ind[k];
            const std::size_t pos = next[static_cast<std::size_t>(r)]++;
            csr.col_ind[pos] = coo_in.col_ind[k];
            csr.values[pos] = coo_in.values[k];
        }

        return csr;
    }

    template <typename T>
    ichol::matrix::CscMatrix<T> coo_to_csc(const ichol::matrix::CooMatrix<T> &coo_in)
    {
        // Classic COO->CSC: count cols, prefix-sum, then scatter entries.
        ichol::matrix::CscMatrix<T> csc;
        csc.num_rows = coo_in.num_rows;
        csc.num_cols = coo_in.num_cols;
        csc.nnz = coo_in.nnz;

        const std::size_t n = static_cast<std::size_t>(csc.num_cols);
        const std::size_t nnz = static_cast<std::size_t>(csc.nnz);

        csc.col_ptr.assign(n + 1, 0);
        for (std::size_t k = 0; k < nnz; ++k)
            ++csc.col_ptr[static_cast<std::size_t>(coo_in.col_ind[k]) + 1];
        for (std::size_t j = 0; j < n; ++j)
            csc.col_ptr[j + 1] += csc.col_ptr[j];

        csc.row_ind.resize(nnz);
        csc.values.resize(nnz);

        auto next = csc.col_ptr;
        for (std::size_t k = 0; k < nnz; ++k)
        {
            const int r = coo_in.row_ind[k];
            const int c = coo_in.col_ind[k];
            const std::size_t pos = next[static_cast<std::size_t>(c)]++;
            csc.row_ind[pos] = r;
            csc.values[pos] = coo_in.values[k];
        }

        // Sort rows within each column to keep canonical CSC.
        for (int cj = 0; cj < csc.num_cols; ++cj)
        {
            const std::size_t j = static_cast<std::size_t>(cj);
            const std::size_t b = static_cast<std::size_t>(csc.col_ptr[j]);
            const std::size_t e = static_cast<std::size_t>(csc.col_ptr[j + 1]);
            const std::size_t len = e - b;
            if (len <= 1)
                continue;

            std::vector<std::size_t> perm(len);
            std::iota(perm.begin(), perm.end(), 0);

            std::sort(perm.begin(), perm.end(),
                      [&](std::size_t a, std::size_t d)
                      {
                          return csc.row_ind[b + a] < csc.row_ind[b + d];
                      });

            std::vector<int> rtmp;
            std::vector<T> vtmp;
            rtmp.reserve(len);
            vtmp.reserve(len);

            for (std::size_t t : perm)
            {
                rtmp.push_back(csc.row_ind[b + t]);
                vtmp.push_back(csc.values[b + t]);
            }

            for (std::size_t t = 0; t < len; ++t)
            {
                csc.row_ind[b + t] = rtmp[t];
                csc.values[b + t] = vtmp[t];
            }
        }

        return csc;
    }
} // namespace

namespace ichol::io
{
    template <typename T>
    matrix::CsrMatrix<T> mtx_to_csr(const std::string &path, bool verify, double alpha)
    {
        auto coo = mtx_to_coo<T>(path, verify, alpha);
        // For numeric: coo is sorted; for pattern Laplacian: generated sorted.
        return coo_to_csr(coo);
    }

    template <typename T>
    matrix::CscMatrix<T> mtx_to_csc(const std::string &path, bool verify, double alpha)
    {
        auto coo = mtx_to_coo<T>(path, verify, alpha);
        return coo_to_csc(coo);
    }

    template matrix::CsrMatrix<double> mtx_to_csr<double>(const std::string &path, bool verify, double alpha);
    template matrix::CsrMatrix<float> mtx_to_csr<float>(const std::string &path, bool verify, double alpha);
    template matrix::CsrMatrix<half_float::half> mtx_to_csr<half_float::half>(const std::string &path, bool verify, double alpha);

    template matrix::CscMatrix<double> mtx_to_csc<double>(const std::string &path, bool verify, double alpha);
    template matrix::CscMatrix<float> mtx_to_csc<float>(const std::string &path, bool verify, double alpha);
    template matrix::CscMatrix<half_float::half> mtx_to_csc<half_float::half>(const std::string &path, bool verify, double alpha);
} // namespace ichol::io
