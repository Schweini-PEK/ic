file(GLOB_RECURSE ICHOL_SRC CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")

list(FILTER ICHOL_SRC EXCLUDE REGEX ".*/apps/.*")

option(ICHOL_USE_MKL_MPCG "Use the CPU oneMKL MPCG backend instead of the CUDA MPCG backend" OFF)
if (ICHOL_USE_MKL_MPCG)
    list(FILTER ICHOL_SRC EXCLUDE REGEX ".*/backends/CUDA/mpcg\\.cu$")
    list(FILTER ICHOL_SRC EXCLUDE REGEX ".*/backends/CUDA/ir\\.cu$")
else()
    list(FILTER ICHOL_SRC EXCLUDE REGEX ".*/solver/mpcg_mkl\\.cpp$")
endif()

add_library(ichol STATIC ${ICHOL_SRC})

target_include_directories(ichol PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src)

target_compile_options(ichol PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

find_package(CUDAToolkit REQUIRED)

find_package(MPI REQUIRED)

target_link_libraries(ichol PUBLIC
  MPI::MPI_CXX
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(CHOLMOD REQUIRED IMPORTED_TARGET CHOLMOD)

# Prefer the AMD that matches the loaded SuiteSparse module
find_library(AMD_LIB amd
    HINTS "$ENV{EBROOTSUITESPARSE}/lib64" "$ENV{EBROOTSUITESPARSE}/lib"
    REQUIRED
)

pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET petsc)
if (NOT TARGET PkgConfig::PETSC)
  pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
endif()

target_link_libraries(ichol PUBLIC
    PkgConfig::CHOLMOD
    PkgConfig::PETSC
    ${AMD_LIB}
)

find_library(SPQR_CFG_LIB suitesparseconfig
    HINTS "$ENV{EBROOTSUITESPARSE}/lib64" "$ENV{EBROOTSUITESPARSE}/lib"
    REQUIRED
)

target_link_libraries(ichol PUBLIC
    ${SPQR_CFG_LIB}
)

target_link_libraries(ichol PUBLIC
    PkgConfig::CHOLMOD
)

target_link_libraries(ichol PUBLIC
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cusolver
)

find_path(MKL_INCLUDE_DIR mkl.h
    HINTS "$ENV{MKLROOT}/include" "$ENV{EBROOTIMKL}/mkl/2023.2.0/include")
find_library(MKL_RT_LIB mkl_rt
    HINTS "$ENV{MKLROOT}/lib/intel64" "$ENV{EBROOTIMKL}/mkl/2023.2.0/lib/intel64")

if (ICHOL_USE_MKL_MPCG)
    if (NOT MKL_INCLUDE_DIR OR NOT MKL_RT_LIB)
        message(FATAL_ERROR "ICHOL_USE_MKL_MPCG requires Intel oneMKL (mkl.h and mkl_rt)")
    endif()
    target_include_directories(ichol PUBLIC ${MKL_INCLUDE_DIR})
    target_link_libraries(ichol PUBLIC ${MKL_RT_LIB})
    target_compile_definitions(ichol PUBLIC ICHOL_USE_MKL_MPCG=1)
endif()

target_link_libraries(ichol PUBLIC PkgConfig::PETSC)

option(ICHOL_USE_OPENMP "Enable OpenMP parallelism for supernodal numeric" ON)
option(ICHOL_USE_BLAS   "Use BLAS/LAPACK (dpotrf/dtrsm/dsyrk) for supernodal numeric" ON)

if (ICHOL_USE_OPENMP)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(ichol PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(ichol PUBLIC ICHOL_USE_OPENMP=1)
    endif()
endif()

if (ICHOL_USE_BLAS)
    if (MKL_INCLUDE_DIR AND MKL_RT_LIB)
        if (ICHOL_USE_OPENMP)
            set(BLA_VENDOR Intel10_64lp)
        else()
            set(BLA_VENDOR Intel10_64lp_seq)
        endif()
    endif()
    find_package(BLAS)
    find_package(LAPACK)
    if (BLAS_FOUND AND LAPACK_FOUND)
        target_link_libraries(ichol PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
        target_compile_definitions(ichol PUBLIC ICHOL_USE_BLAS=1)
    endif()
endif()

add_executable(ict_pcg
    apps/ict_pcg_main.cpp
)

target_link_libraries(ict_pcg PRIVATE ichol)

add_executable(mpcg_pcg_compare
    apps/mpcg_pcg_compare_main.cpp
)

target_link_libraries(mpcg_pcg_compare PRIVATE ichol)

if (NOT ICHOL_USE_MKL_MPCG)
    if (MKL_INCLUDE_DIR AND MKL_RT_LIB)
        add_library(ichol_mkl_baseline STATIC
            solver/mpcg_mkl.cpp
        )

        target_include_directories(ichol_mkl_baseline PUBLIC
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
            ${MKL_INCLUDE_DIR})

        target_link_libraries(ichol_mkl_baseline PUBLIC
            ichol
            ${MKL_RT_LIB})

        add_executable(mpcg_cpu_baseline
            apps/mpcg_cpu_baseline.cpp
        )

        target_link_libraries(mpcg_cpu_baseline PRIVATE
            ichol
            ichol_mkl_baseline)
    else()
        message(WARNING "Intel oneMKL not found: mpcg_cpu_baseline target will be disabled")
    endif()

    add_executable(mpcg_improve
        apps/mpcg_improve.cpp
    )

    target_link_libraries(mpcg_improve PRIVATE ichol)

    add_executable(mpcg_trunc
        apps/mpcg_trunc.cpp
    )

    target_link_libraries(mpcg_trunc PRIVATE ichol)

    add_executable(heatmap_vis
        apps/heatmap_vis.cpp
    )

    target_link_libraries(heatmap_vis PRIVATE ichol)

    add_executable(mpcg_low
        apps/mpcg_low.cpp
    )

    target_link_libraries(mpcg_low PRIVATE ichol)

    add_executable(ir
        apps/ir.cpp
    )

    target_link_libraries(ir PRIVATE ichol)
endif()

find_path(CUDSS_INCLUDE_DIR cudss.h
    HINTS "$ENV{EBROOTCUDSS}/include" "$ENV{CUDA_HOME}/include" "$ENV{EBROOTCUDA}/include")
find_library(CUDSS_LIB NAMES cudss
    HINTS "$ENV{EBROOTCUDSS}/lib64" "$ENV{EBROOTCUDSS}/lib" "$ENV{CUDA_HOME}/lib64" "$ENV{CUDA_HOME}/lib" "$ENV{EBROOTCUDA}/lib64" "$ENV{EBROOTCUDA}/lib")

if(CUDSS_INCLUDE_DIR AND CUDSS_LIB)
    target_include_directories(ichol PRIVATE ${CUDSS_INCLUDE_DIR})
    target_link_libraries(ichol PUBLIC ${CUDSS_LIB})
    target_compile_definitions(ichol PUBLIC ICHOL_HAVE_CUDSS=1)
else()
    message(WARNING "cuDSS not found: exact subdomain preconditioner backend will be disabled")
    target_compile_definitions(ichol PUBLIC ICHOL_HAVE_CUDSS=0)
endif()
