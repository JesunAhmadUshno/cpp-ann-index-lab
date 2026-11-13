cmake_minimum_required(VERSION 3.20)
project(cpp-ann-index-lab CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(ann INTERFACE)
target_include_directories(ann INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(ann INTERFACE -O3 -Wall -Wextra)

add_executable(ann_bench src/ann_bench.cpp)
target_link_libraries(ann_bench PRIVATE ann)

enable_testing()
add_executable(ann_tests tests/test_ann.cpp)
target_link_libraries(ann_tests PRIVATE ann)
add_test(NAME ann_tests COMMAND ann_tests)
