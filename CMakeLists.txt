cmake_minimum_required(VERSION 3.25)
project(kascade LANGUAGES CXX)
option(KASCADE_BUILD_BENCHMARKS "Build benchmarks" ${PROJECT_IS_TOP_LEVEL})

find_package(MPI REQUIRED)

include(FetchContent)
if(PROJECT_IS_TOP_LEVEL)
  FetchContent_Declare(
    Format.cmake
    GIT_REPOSITORY https://github.com/TheLartians/Format.cmake
    GIT_TAG v1.8.3)
  FetchContent_MakeAvailable(Format.cmake)
endif()

add_library(${CMAKE_PROJECT_NAME}_cxx_flags INTERFACE)
if(KASCADE_CXX_FLAGS)
  target_compile_options(${CMAKE_PROJECT_NAME}_cxx_flags
                         INTERFACE ${KASCADE_CXX_FLAGS})
endif()

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG 11.0.2
  GIT_SHALLOW ON
  SYSTEM FIND_PACKAGE_ARGS 11.0)
set(SPDLOG_FMT_EXTERNAL_HO ON)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog
  GIT_TAG v1.16.0
  GIT_SHALLOW ON
  SYSTEM)
FetchContent_MakeAvailable(fmt spdlog)

FetchContent_Declare(
  kassert
  GIT_REPOSITORY https://github.com/kamping-site/kassert
  GIT_TAG v1.0.0
  SYSTEM EXCLUDE_FROM_ALL)

FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping
  GIT_TAG v0.2.0
  SYSTEM EXCLUDE_FROM_ALL)

set(KAGEN_BUILD_APPS OFF)
set(KAGEN_BUILD_EXAMPLES OFF)
set(KAGEN_USE_CGAL OFF)
set(KAGEN_USE_SPARSEHASH OFF)
set(KAGEN_USE_XXHASH ON)
FetchContent_Declare(
  kagen
  GIT_REPOSITORY https://github.com/KarlsruheGraphGeneration/KaGen
  GIT_TAG main
  GIT_SHALLOW ON
  EXCLUDE_FROM_ALL SYSTEM)

FetchContent_MakeAvailable(kamping kassert kagen)
add_subdirectory(lib)

if(KASCADE_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

FetchContent_Declare(
  kaval
  GIT_REPOSITORY https://github.com/niklas-uhl/kaval.git
  GIT_TAG main
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/kaval)
FetchContent_MakeAvailable(kaval)
