add_subdirectory(graph)

add_library(${CMAKE_PROJECT_NAME}_list_ranking)
target_sources(
  ${CMAKE_PROJECT_NAME}_list_ranking
  PRIVATE list_ranking.cpp
          local_pointer_chasing.cpp
          successor_utils.cpp
          pointer_doubling.cpp
          async_pointer_doubling.cpp
          rma_pointer_doubling.cpp
          grid_communicator.cpp
          eulertour.cpp
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${CMAKE_CURRENT_LIST_DIR}/..
         FILES
         grid_communicator.hpp
         list_ranking.hpp
         successor_utils.hpp
         pointer_doubling.hpp
         configuration.hpp
         eulertour.hpp
         types.hpp)
add_subdirectory(sparse_ruling_set_detail)
target_link_libraries(
  ${CMAKE_PROJECT_NAME}_list_ranking
  PUBLIC MPI::MPI_CXX kamping::kamping ${CMAKE_PROJECT_NAME}_graph
  PRIVATE ${CMAKE_PROJECT_NAME}_cxx_flags BriefKAsten::BriefKAsten
          kassert::kassert fmt::fmt spdlog::spdlog absl::flat_hash_map)
target_compile_features(${CMAKE_PROJECT_NAME}_list_ranking PUBLIC cxx_std_20)
