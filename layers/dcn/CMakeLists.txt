cmake_minimum_required(VERSION 3.8)
project(deformable_conv)

set(CMAKE_C_COMPILER /usr/bin/gcc-4.8)
set(CMAKE_CXX_COMPILER /usr/bin/g++-4.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -ftest-coverage -fprofile-arcs")
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_FLAGS -D_GLIBCXX_USE_CXX11_ABI=0)


# set(TF_INCLUDE_PATH /home/dl/anaconda3/envs/tf18/lib/python3.6/site-packages/tensorflow/include)
set(TF_INCLUDE_PATH /data/lupu/miniconda3/envs/tf18/lib/python3.6/site-packages/tensorflow/include)
# set(TF_INCLUDE_PATH /home/dl/anaconda3/envs/tf18/lib/python3.6/site-packages/tensorflow/include/third_party/gpus)

# include_directories(/home/lupu/custom_ops/lib/include)
# include_directories(/data/lupu/miniconda3/envs/tf18/lib/python3.6/site-packages/tensorflow/include/third_party/eigen3)

#add_definitions(-D__JETBRAINS_IDE__)
add_definitions(-DEIGEN_USE_THREADS)
# option(USE_CUDA OFF)
set(USE_CUDA ON)

if (USE_CUDA)
    add_definitions(-DGOOGLE_CUDA=1)
	add_definitions(-DEIGEN_USE_GPU)
	find_package(CUDA REQUIRED)

	enable_language(CUDA)
	set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
	set(CMAKE_CUDA_FLAGS "-std=c++11 --expt-relaxed-constexpr")
endif ()

add_library(ops SHARED deformable_conv2d.cpp deformable_conv.cu  deformable_conv2d.h deformable_conv2d_utils.h)
target_include_directories(ops PUBLIC ${CUDA_INCLUDE_DIRS} ${TF_INCLUDE_PATH})
message(STATUS ${TF_INCLUDE_PATH})

#set_property(TARGET ops PROPERTY CXX_STANDARD 11)
# target_link_libraries(ops PUBLIC ${CUDA_LIBRARIES} /home/dl/anaconda3/envs/tf18/lib/python3.6/site-packages/tensorflow/libtensorflow_framework.so.1)
target_link_libraries(ops PUBLIC ${CUDA_LIBRARIES} /data/lupu/miniconda3/envs/tf18/lib/python3.6/site-packages/tensorflow/libtensorflow_framework.so.1)
target_compile_definitions(ops PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)