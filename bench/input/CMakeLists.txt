add_library(${CMAKE_PROJECT_NAME}_input_generation)
target_sources(
  ${CMAKE_PROJECT_NAME}_input_generation
  PRIVATE graph.cpp bfs.cpp bfs_forest_generation.cpp generation.cpp
          no_preprocessing.cpp
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ../
         FILES
         generation.hpp
         graph.hpp
         bfs.hpp
         concepts.hpp)
target_link_libraries(
  ${CMAKE_PROJECT_NAME}_input_generation
  PRIVATE ${CMAKE_PROJECT_NAME}_cxx_flags
  PUBLIC kamping::kamping
         cmake_git_version_tracking
         KaGen::KaGen
         absl::flat_hash_map
         fmt::fmt
         spdlog::spdlog
         kassert::kassert
         kascade_list_ranking)

target_compile_features(${CMAKE_PROJECT_NAME}_input_generation
                        PUBLIC cxx_std_23)
