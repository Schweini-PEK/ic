#pragma once

/**
 * @file snode_schedule.hpp
 * @brief Backward-compatible include shim.
 *
 * @details
 * The real schedule helpers and `SupernodalLLPlan` now live in
 * `supernodal_ll_plan.hpp`. This file intentionally re-exports that header
 * so old include paths keep compiling.
 *
 * @par Problems
 * - Severity: Low
 * - Issue: Long-lived shims can hide include hygiene issues and stale deps.
 * - Suggested fix: Migrate call sites to include `supernodal_ll_plan.hpp`
 *   directly, then remove this shim in a compatibility-breaking cleanup.
 */
#include "factor/symbolic/supernodal_ll_plan.hpp"
