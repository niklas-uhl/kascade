FetchContent_Declare(
  cmake_git_version_tracking
  GIT_REPOSITORY https://github.com/niklas-uhl/cmake-git-version-tracking
  GIT_TAG master)
FetchContent_MakeAvailable(cmake_git_version_tracking)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG v3.12.0
  GIT_SHALLOW ON
  SYSTEM)
FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11
  GIT_TAG v2.5.0
  GIT_SHALLOW ON
  SYSTEM)
FetchContent_Declare(
  kamping-spdlog-adapter
  GIT_REPOSITORY https://github.com/kamping-site/kamping-spdlog-adapter
  GIT_TAG main
  GIT_SHALLOW ON)
FetchContent_Declare(
  kamping-nlohmann-json-adapter
  GIT_REPOSITORY https://github.com/kamping-site/kamping-nlohmann-json-adaptor
  GIT_TAG 9685077
  GIT_SHALLOW ON)
set(ABSL_PROPAGATE_CXX_STD ON)
FetchContent_Declare(
  absl
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp
  GIT_TAG 20250814.0
  GIT_SHALLOW ON
  PATCH_COMMAND
    patch -p1 --forward <
    ${PROJECT_SOURCE_DIR}/cmake/patches/macos_no_define_xopen_source.patch ||
    true EXCLUDE_FROM_ALL SYSTEM)

FetchContent_MakeAvailable(nlohmann_json cli11 kamping-spdlog-adapter
                           kamping-nlohmann-json-adapter absl)

add_subdirectory(input)

add_executable(kascade)

target_sources(
  kascade
  PRIVATE runner.cpp
          detail/parsing.cpp
          detail/algorithm_factory.cpp
          detail/reporting.cpp
          detail/verification.cpp
          detail/stats.cpp
          detail/mplr/mplr.cpp
  PRIVATE FILE_SET
          HEADERS
          BASE_DIRS
          ${CMAKE_CURRENT_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}
          FILES
          detail/mplr/mplr.hpp
          detail/mplr/communicator.hpp
          detail/mplr/forest_irregular_pointer_doubling.hpp
          detail/mplr/forest_regular_ruling_set2.hpp
          detail/mplr/helper_functions.hpp
          detail/benchmark_config.hpp
          detail/verification.hpp
          detail/parsing.hpp
          detail/serialization.hpp
          detail/algorithm.hpp
          detail/algorithm_factory.hpp
          detail/algorithm_impl.hpp
          detail/stats.hpp
          detail/reporting.hpp)

target_link_libraries(
  kascade
  PRIVATE ${CMAKE_PROJECT_NAME}_cxx_flags
          cmake_git_version_tracking
          KaGen::KaGen
          CLI11::CLI11
          nlohmann_json::nlohmann_json
          kamping::spdlog-adapter
          kamping::nlohmann-json-adapter
          kassert::kassert
          kascade_list_ranking
          kascade_input_generation)

target_compile_features(kascade PRIVATE cxx_std_23)
