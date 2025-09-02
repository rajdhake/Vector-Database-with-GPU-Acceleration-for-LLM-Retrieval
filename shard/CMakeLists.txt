cmake_minimum_required(VERSION 3.18)
project(vdb LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_library(vdb MODULE src/shard_binding.cpp src/cuda/flat_kernels.cu)

# Adjust your GPU arch as needed
set_target_properties(vdb PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;86;89"
)

# On some systems pybind11 adds correct flags automatically
# Otherwise ensure -fPIC, and proper extension suffix

# Include dirs
target_include_directories(vdb PRIVATE include ${Python3_INCLUDE_DIRS})

# Define module name properly for Python
if (WIN32)
    set_target_properties(vdb PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(vdb PROPERTIES PREFIX "" SUFFIX ".so")
endif()

# Link
target_link_libraries(vdb PRIVATE pybind11::module)