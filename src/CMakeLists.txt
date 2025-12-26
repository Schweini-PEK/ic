file(GLOB_RECURSE ICHOL_SRC
     "${CMAKE_CURRENT_SOURCE_DIR}/backends/*.cu"
     "${CMAKE_CURRENT_SOURCE_DIR}/backends/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cu"
     "${CMAKE_CURRENT_SOURCE_DIR}/io/*.cpp")

add_library(ichol STATIC ${ICHOL_SRC}
        ../include/ichol/supernode.hpp)

target_include_directories(ichol PUBLIC
    ${PROJECT_SOURCE_DIR}/include)

target_compile_options(ichol PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(ichol PUBLIC
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
)

add_executable(ict_pcg
    apps/ict_pcg_main.cpp
)

target_link_libraries(ict_pcg PRIVATE ichol)