#pragma once

#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/json.hpp>

#include "benchmark_config.hpp"
#include "detail/stats.hpp"
#include "input/generation.hpp"
#include "kascade/configuration.hpp"

NLOHMANN_JSON_SERIALIZE_ENUM(StatsLevel,
                             {{StatsLevel::invalid, nullptr},
                              {StatsLevel::none, "none"},
                              {StatsLevel::basic, "basic"},
                              {StatsLevel::extensive, "extensive"},
                              {StatsLevel::reduced_extensive,
                               "extensive-reduced-output"}})

namespace kascade::input {
NLOHMANN_JSON_SERIALIZE_ENUM(InputProcessing,
                             {{InputProcessing::invalid, nullptr},
                              {InputProcessing::bfs, "bfs"},
                              {InputProcessing::none, "none"}})

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Config, kagen_option_string, input_processing);

}  // namespace kascade::input

namespace kascade {
NLOHMANN_JSON_SERIALIZE_ENUM(Algorithm,
                             {{Algorithm::invalid, nullptr},
                              {Algorithm::GatherChase, "GatherChase"},
                              {Algorithm::PointerDoubling, "PointerDoubling"},
                              {Algorithm::AsyncPointerDoubling, "AsyncPointerDoubling"},
                              {Algorithm::RMAPointerDoubling, "RMAPointerDoubling"},
                              {Algorithm::EulerTour, "EulerTour"},
                              {Algorithm::MPLR, "MPLR"},
                              {Algorithm::SparseRulingSet, "SparseRulingSet"}})
NLOHMANN_JSON_SERIALIZE_ENUM(AggregationLevel,
                             {{AggregationLevel::invalid, nullptr},
                              {AggregationLevel::none, "none"},
                              {AggregationLevel::local, "local"},
                              {AggregationLevel::all, "all"}})

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AsyncPointerChasingConfig, use_caching);
NLOHMANN_JSON_SERIALIZE_ENUM(RMASyncMode,
                             {{RMASyncMode::invalid, nullptr},
                              {RMASyncMode::fenced, "fenced"},
                              {RMASyncMode::passive_target, "passive_target"}});
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RMAPointerChasingConfig, sync_mode, batch_size);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PointerDoublingConfig,
                                   use_local_preprocessing,
                                   aggregation_level);
NLOHMANN_JSON_SERIALIZE_ENUM(RulerSelectionStrategy,
                             {{RulerSelectionStrategy::invalid, nullptr},
                              {RulerSelectionStrategy::dehne, "dehne"},
                              {RulerSelectionStrategy::heuristic, "heuristic"},
                              {RulerSelectionStrategy::sanders, "sanders"}});
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BriefkastenConfig,
                                   local_threshold,
                                   poll_skip_threshold);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SparseRulingSetConfig,
                                   ruler_selection,
                                   dehne_factor,
                                   heuristic_factor,
                                   sanders_factor,
                                   sync,
                                   sync_locality_aware,
                                   spawn,
                                   briefkasten,
                                   use_grid_communication);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(EulerTourConfig, algorithm, use_high_degree_handling);
}  // namespace kascade

namespace mplr {
NLOHMANN_JSON_SERIALIZE_ENUM(Algorithm,
                             {{Algorithm::invalid, nullptr},
                              {Algorithm::PointerDoubling, "PointerDoubling"},
                              {Algorithm::ForestRulingSet, "ForestRulingSet"}})
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Configuration,
                                   algorithm,
                                   comm_rounds,
                                   recursion_levels,
                                   use_grid,
                                   use_aggregation);
}  // namespace mplr

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(TreeStats, size, max_rank, rank_sum);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(BasicStats,
                                                  num_trees,
                                                  max_rank,
                                                  avg_rank)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(ExtensiveStats,
                                                  // num_trees,
                                                  num_nontrivial_trees,
                                                  // nontrivial_size_sum,
                                                  max_size,
                                                  avg_size,
                                                  nontrivial_avg_size,
                                                  nontrivial_avg_rank,
                                                  per_tree_stats)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(Stats, basic_stats, extensive_stats)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Config,
                                   num_ranks,
                                   git_tag,
                                   output_path,
                                   iterations,
                                   input,
                                   algorithm,
                                   async_pointer_chasing,
                                   rma_pointer_chasing,
                                   pointer_doubling,
                                   sparse_ruling_set,
                                   euler_tour,
                                   mplr,
                                   verify_level,
                                   verify_continue_on_mismatch,
                                   statistics_level)
