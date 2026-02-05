add_library(${CMAKE_PROJECT_NAME}_graph)
target_sources(
  ${CMAKE_PROJECT_NAME}_graph
  PRIVATE graph.cpp
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ../
         FILES
         graph.hpp
         concepts.hpp)
target_link_libraries(
  ${CMAKE_PROJECT_NAME}_graph
  PRIVATE ${CMAKE_PROJECT_NAME}_cxx_flags
  PUBLIC kamping::kamping
         cmake_git_version_tracking
         KaGen::KaGen
         absl::flat_hash_map
         fmt::fmt
         spdlog::spdlog
         kassert::kassert)

target_compile_features(${CMAKE_PROJECT_NAME}_graph
                        PUBLIC cxx_std_23)
