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
FetchContent_MakeAvailable(nlohmann_json cli11 kamping-spdlog-adapter)

add_executable(kascade)

target_sources(
  kascade
  PRIVATE runner.cpp detail/generation.cpp detail/parsing.cpp
  # detail/reporting.cpp detail/algorithm_factory.cpp
  PRIVATE FILE_SET
          HEADERS
          BASE_DIRS
          ${CMAKE_CURRENT_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}
          FILES
          detail/benchmark_config.hpp
          detail/generation.hpp
          detail/parsing.hpp
          detail/serialization.hpp
  # detail/reporting.hpp detail/algorithm.hpp detail/algorithm_impl.hpp
  # detail/algorithm_factory.hpp ${CMAKE_CURRENT_BINARY_DIR}/detail/config.h)
)

target_link_libraries(
  kascade PRIVATE cmake_git_version_tracking KaGen::KaGen CLI11::CLI11
                  nlohmann_json::nlohmann_json kamping::spdlog-adapter kassert::kassert kascade_list_ranking)

target_compile_features(kascade PRIVATE cxx_std_23)
