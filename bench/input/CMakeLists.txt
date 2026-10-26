add_library(${CMAKE_PROJECT_NAME}_input_generation)
target_sources(
  ${CMAKE_PROJECT_NAME}_input_generation
  PRIVATE bfs.cpp bfs_forest_generation.cpp generation.cpp no_preprocessing.cpp
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ../
         FILES
         generation.hpp
         bfs.hpp)
target_link_libraries(
  ${CMAKE_PROJECT_NAME}_input_generation
  PRIVATE ${CMAKE_PROJECT_NAME}_cxx_flags
  PUBLIC absl::flat_hash_map
         kamping::kamping
         KaGen::KaGen
         fmt::fmt
         spdlog::spdlog
         kassert::kassert
         kascade_list_ranking
         kascade_graph)

target_compile_features(${CMAKE_PROJECT_NAME}_input_generation
                        PUBLIC cxx_std_23)
