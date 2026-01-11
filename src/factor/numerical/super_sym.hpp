#pragma once
#include <vector>

namespace ichol::symbolic
{
    struct SuperSym
    {
        std::vector<int> super; // nsuper+1
        std::vector<int> pi;    // nsuper+1
        std::vector<int> px;    // nsuper+1
        std::vector<int> s;     // pi.back()
    };
    SuperSym build_super_sym(
    const std::vector<std::pair<int,int>>& snodes,
    const std::vector<std::vector<int>>& snode_rows);
}
