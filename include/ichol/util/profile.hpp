// include/ichol/util/profile.hpp

#pragma once
#ifdef ICHOL_PROFILE
#include "timer.hpp"
#define ICHOL_SCOPED_TIMER(NAME, SINK) ::ichol::util::ScopedTimer _t__(NAME, SINK)
#else
#define ICHOL_SCOPED_TIMER(NAME, SINK) \
    do                                 \
    {                                  \
    } while (0)
#endif