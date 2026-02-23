target_sources(
  ${CMAKE_PROJECT_NAME}_list_ranking
  PRIVATE FILE_SET
          ${CMAKE_PROJECT_NAME}_list_ranking_sparse_ruling_set_detail
          TYPE
          HEADERS
          BASE_DIRS
          ${CMAKE_CURRENT_LIST_DIR}/../..
          FILES
          ruler_chasing_engine.hpp
          ruler_propagation.hpp
	  ruler_selection.hpp
          types.hpp
	  trace.hpp
	)
