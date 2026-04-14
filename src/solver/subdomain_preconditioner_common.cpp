#include "ichol/subdomain_preconditioner_common.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ichol::precond
{
    std::vector<SubdomainRegion> partition_subdomains(
        const GridShape &global_shape,
        const SubdomainSize &sub_size)
    {
        if (global_shape.w <= 0 || global_shape.h <= 0 || global_shape.d <= 0)
            throw std::runtime_error("partition_subdomains: invalid global shape");
        if (sub_size.w <= 0 || sub_size.h <= 0 || sub_size.d <= 0)
            throw std::runtime_error("partition_subdomains: invalid subdomain size");

        std::vector<SubdomainRegion> regions;
        for (int z = 0; z < global_shape.d; z += sub_size.d)
        {
            for (int y = 0; y < global_shape.h; y += sub_size.h)
            {
                for (int x = 0; x < global_shape.w; x += sub_size.w)
                {
                    regions.push_back({x, std::min(x + sub_size.w, global_shape.w),
                                       y, std::min(y + sub_size.h, global_shape.h),
                                       z, std::min(z + sub_size.d, global_shape.d)});
                }
            }
        }
        return regions;
    }
} // namespace ichol::precond
