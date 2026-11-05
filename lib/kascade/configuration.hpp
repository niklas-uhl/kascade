#pragma once

#include <any>
#include <cstddef>
#include <cstdint>
#include <string>

namespace kascade {

enum class Algorithm : std::uint8_t {
  GatherChase,
  PointerDoubling,
  AsyncPointerDoubling,
  RMAPointerDoubling,
  SparseRulingSet,
  EulerTour,
  MPLR,
  invalid,
};

enum class AggregationLevel : std::uint8_t { none, local, all, invalid };

struct AsyncPointerChasingConfig {
  bool use_caching = false;
};

enum class RMASyncMode : std::uint8_t { passive_target, fenced, invalid };

struct RMAPointerChasingConfig {
  RMASyncMode sync_mode = RMASyncMode::passive_target;
  std::size_t batch_size = 1;
};

enum class GridCommunicatorMode : std::uint8_t {
  topology_aware,
  balanced,
  invalid
};

struct PointerDoublingConfig {
  bool use_local_preprocessing = true;
  GridCommunicatorMode grid_communicator_mode = GridCommunicatorMode::topology_aware;
  bool use_grid_communication = false;
  bool use_local_first_request_scheme = true;
  bool cache_succ_owners = true;
  AggregationLevel aggregation_level = AggregationLevel::none;
  double fallback_allgather_size_ratio = 0.0;
};

enum class RulerSelectionStrategy : std::uint8_t {
  dehne,
  heuristic,
  sanders,
  ultimate,
  ultimate_sanders,
  limit_rounds,
  invalid
};
enum class RulerPropagationMode : std::uint8_t { pull, push, invalid };

// How the sparse ruling set's personalized exchanges are performed:
// alltoallv: MPI_Alltoallv, preceded by an MPI_Alltoall of the counts
// nbx:       kamping's sparse alltoall (NBX, Hoefler et al. 2010)
// pcx:       personalized census (PCX, Hoefler et al. 2010)
enum class AlltoallAlgorithm : std::uint8_t { alltoallv, nbx, pcx, invalid };

// What briefkasten's terminate() flushes when the local work queue runs empty
// (briefkasten::IdleFlush). library_default: all for the flat queue, off under grid
// indirection.
enum class IdleFlush : std::uint8_t { library_default, off, all, stale, invalid };

struct BriefkastenConfig {
  std::size_t local_threshold = 2048;
  std::size_t poll_skip_threshold = 1000;
  IdleFlush idle_flush = IdleFlush::library_default;
  // Restart every idle flush at the beginning of the buffer map instead of where the previous one
  // ran out of send slots (briefkasten's Config::idle_flush_round_robin = false). A/B only.
  bool idle_flush_fixed_start = false;
  // Send backlog behind the MPI request slots: "default" (briefkasten's choice, 0 under flow
  // control), "comm-size", or a number. An explicit value survives flow control.
  std::string send_backlog_capacity = "default";
};

struct SparseRulingSetConfig {
  RulerSelectionStrategy ruler_selection = RulerSelectionStrategy::sanders;
  Algorithm base_algorithm = Algorithm::PointerDoubling;
  std::any base_algorithm_config;
  std::size_t current_sparse_ruling_set_round = 0;
  std::size_t sparse_ruling_set_rounds = 1;
  RulerPropagationMode ruler_propagation_mode = RulerPropagationMode::pull;
  bool use_aggregation_in_ruler_propagation = false;
  bool use_local_contraction = false;
  bool reverse_list_locality_aware = true;
  bool use_locality_aware_in_ruler_propagation = true;
  double dehne_factor = 1.0;
  double heuristic_factor = 0.01;
  double sanders_factor = 1.0;
  double ultimate_factor = 0.1;
  double ultimate_sanders_factor = 2.0;
  std::size_t round_limit = 100;
  bool cache_owners = false;
  std::size_t root_gather_threshold = 100;
  bool sync = true;
  bool sync_locality_aware = false;
  bool spawn = false;
  bool no_precompute_rulers = false;
  bool post_invert = false;
  bool post_invert_detect_leaves = false;
  BriefkastenConfig briefkasten{};
  GridCommunicatorMode grid_communicator_mode = GridCommunicatorMode::topology_aware;
  bool use_grid_communication = false;
  bool use_local_first_request_scheme = false;
  AlltoallAlgorithm alltoall_algorithm = AlltoallAlgorithm::alltoallv;
};

struct EulerTourConfig {
  Algorithm algorithm;
  bool use_high_degree_handling = false;
  std::any algo_config;
  GridCommunicatorMode grid_communicator_mode = GridCommunicatorMode::topology_aware;
  bool use_grid_communication = false;
};
}  // namespace kascade
