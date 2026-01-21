file(GLOB_RECURSE ICHOL_SRC
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")

list(FILTER ICHOL_SRC EXCLUDE REGEX ".*/apps/.*")

add_library(ichol STATIC ${ICHOL_SRC}
        ../include/ichol/supernode.hpp
)
target_include_directories(ichol PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src)

target_compile_options(ichol PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(ichol PRIVATE CUDA::cusolver CUDA::cudart)

target_link_libraries(ichol PUBLIC
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
)

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

