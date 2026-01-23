add_library(${CMAKE_PROJECT_NAME}_list_ranking)
target_sources(
  ${CMAKE_PROJECT_NAME}_list_ranking
  PRIVATE list_ranking.cpp pointer_doubling.cpp async_pointer_doubling.cpp
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${CMAKE_CURRENT_LIST_DIR}/..
         FILES
         list_ranking.hpp
         pointer_doubling.hpp
	 configuration.hpp
         types.hpp)
target_link_libraries(
  ${CMAKE_PROJECT_NAME}_list_ranking
  PUBLIC
  MPI::MPI_CXX
  kamping::kamping
  PRIVATE
  ${CMAKE_PROJECT_NAME}_cxx_flags
  BriefKAsten::BriefKAsten
  kassert::kassert
  fmt::fmt
  spdlog::spdlog
  absl::flat_hash_map
)
target_compile_features(${CMAKE_PROJECT_NAME}_list_ranking PUBLIC cxx_std_20)
