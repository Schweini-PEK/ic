// include/ichol/util/timer.hpp
#pragma once
#include <chrono>
#include <string_view>
#include <functional>

namespace ichol::util
{

    using TimeSink = std::function<void(std::string_view, double)>;

    struct ScopedTimer
    {
        std::string_view name;
        TimeSink sink;
        std::chrono::steady_clock::time_point t0;

        ScopedTimer(std::string_view n, TimeSink s)
            : name(n), sink(std::move(s)), t0(std::chrono::steady_clock::now()) {}

        ~ScopedTimer()
        {
            if (!sink)
                return;
            auto t1 = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            sink(name, sec);
        }
    };

} // namespace ichol::util
