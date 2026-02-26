#include "kascade/sparse_ruling_set.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <ranges>
#include <utility>

#include <absl/container/flat_hash_map.h>
#include <kamping/communicator.hpp>
#include <kamping/measurements/timer.hpp>
#include <kassert/kassert.hpp>
#include <spdlog/spdlog.h>

#include "kascade/assertion_levels.hpp"
#include "kascade/bits.hpp"
#include "kascade/configuration.hpp"
#include "kascade/distribution.hpp"
#include "kascade/grid_communicator.hpp"
#include "kascade/list_ranking.hpp"
#include "kascade/pack.hpp"
#include "kascade/pointer_doubling.hpp"
#include "kascade/successor_utils.hpp"
#include "kascade/types.hpp"
#include "sparse_ruling_set_detail/post_invert.hpp"
#include "sparse_ruling_set_detail/ruler_chasing_engine.hpp"
#include "sparse_ruling_set_detail/ruler_propagation.hpp"
#include "sparse_ruling_set_detail/ruler_selection.hpp"
#include "sparse_ruling_set_detail/trace.hpp"
#include "sparse_ruling_set_detail/types.hpp"

namespace kascade {
namespace sparse_ruling_set_detail {

using BaseAlgorithm = std::function<void(std::span<idx_t>,
                                         std::span<rank_t>,
                                         Distribution const&,
                                         kamping::Communicator<> const&)>;
namespace {

void sparse_ruling_set(SparseRulingSetConfig const& config,
                       std::span<idx_t> succ_array,
                       std::span<rank_t> rank_array,
                       Distribution const& dist,
                       BaseAlgorithm const& base_algorithm,
                       kamping::Communicator<> const& comm) {
  KASSERT(is_list(succ_array, dist, comm), assert::with_communication);
  kamping::measurements::timer().start("init_grid_comm");
  std::optional<TopologyAwareGridCommunicator> grid_comm;
  if (config.use_grid_communication) {
    grid_comm = TopologyAwareGridCommunicator{comm};
  }
  kamping::measurements::timer().stop();
  std::vector<idx_t> leaves;
  kamping::measurements::timer().synchronize_and_start("invert_list");
  if (!config.post_invert) {
    leaves = reverse_list(succ_array, rank_array, succ_array, rank_array, dist, comm,
                          grid_comm, config.use_grid_communication,
                          config.reverse_list_locality_aware);
  }
  kamping::measurements::timer().stop();
  kamping::measurements::timer().synchronize_and_start("find_leaves");
  if (config.post_invert) {
    leaves = LeafInfo{succ_array, dist, comm}.leaves() | std::ranges::to<std::vector>();
  }
  kamping::measurements::timer().stop();

  kamping::measurements::timer().start("cache_owners");
  std::optional<std::vector<std::size_t>> succ_owner;
  if (config.cache_owners) {
    succ_owner.emplace(succ_array.size());
    for (std::size_t i = 0; i < succ_owner->size(); i++) {
      auto succ = succ_array[i];
      if (dist.is_local(succ, comm.rank())) {
        (*succ_owner)[i] = comm.rank();
      } else {
        (*succ_owner)[i] = dist.get_owner(succ);
      }
    }
  }
  kamping::measurements::timer().stop();
  auto get_succ_owner = [&](idx_t local_idx, idx_t succ_global) {
    if (!config.cache_owners) {
      return dist.get_owner(succ_global);
    }
    KASSERT(dist.get_owner(succ_global) == (*succ_owner)[local_idx]);
    return (*succ_owner)[local_idx];
  };

  kamping::measurements::timer().start("init_node_type");
  std::vector<NodeType> node_type(succ_array.size(), NodeType::unreached);
  std::size_t num_unreached = 0;
  std::vector<idx_t> local_indices_permuted;
  std::mt19937 rng{static_cast<std::mt19937::result_type>(42 + comm.rank_signed())};

  if (!config.no_precompute_rulers) {
    local_indices_permuted =
        dist.local_indices(comm.rank()) | std::ranges::to<std::vector>();
    std::ranges::shuffle(local_indices_permuted, rng);
  }

  for (auto local_idx : dist.local_indices(comm.rank())) {
    if (is_root(local_idx, succ_array, dist, comm)) {
      node_type[local_idx] = NodeType::root;
    } else {
      num_unreached++;
    }
  }
  std::size_t num_real_leaves = 0;
  for (auto leaf_local : leaves) {
    if (node_type[leaf_local] == NodeType::root) {
      continue;
    }
    node_type[leaf_local] = NodeType::leaf;
    num_real_leaves++;
    num_unreached--;
  }
  kamping::measurements::timer().stop();
  kamping::measurements::timer().start("find_rulers");

  std::int64_t local_num_rulers =
      // NOLINTNEXTLINE(*-narrowing-conversions)
      static_cast<std::int64_t>(compute_local_num_rulers(config, dist, comm)) -
      num_real_leaves;
  local_num_rulers = std::max(local_num_rulers, std::int64_t{0});

  RulerTrace trace{static_cast<size_t>(local_num_rulers), num_real_leaves};

  SPDLOG_DEBUG("picking {} rulers", local_num_rulers);

  auto [rulers, next_ruler_it] = pick_rulers(
      config, local_num_rulers, dist, rng, local_indices_permuted,
      [&](idx_t local_idx) { return node_type[local_idx] == NodeType::unreached; }, comm);
  num_unreached -= rulers.size();
  kamping::measurements::timer().stop();

  //////////////////////
  // Ruler chasing    //
  //////////////////////
  kamping::measurements::timer().synchronize_and_start("backup_succ_array");
  std::optional<std::vector<idx_t>> inital_succ_array;
  if (config.ruler_propagation_mode == RulerPropagationMode::push) {
    inital_succ_array = std::vector(succ_array.begin(), succ_array.end());
  }
  kamping::measurements::timer().stop();
  kamping::measurements::timer().synchronize_and_start("chase_ruler");

  // initialization
  for (auto leaf_local : leaves) {
    // only use "real" leaves as rulers
    if (node_type[leaf_local] == NodeType::leaf) {
      rulers.push_back(leaf_local);
    }
  }
  // chasing loop
  auto init = [&](auto&& enqueue_locally, auto&& send_to) {
    for (auto const& ruler_local : rulers) {
      auto ruler = dist.get_global_idx(ruler_local, comm.rank());
      auto succ = succ_array[ruler_local];
      auto dist_to_succ = rank_array[ruler_local];
      if (node_type[ruler_local] == NodeType::leaf) {
        // this is a leaf, so set the msb to distinguish it from normal rulers, which
        // will help to avoid them requesting their ruler's information in the end
        if (config.ruler_propagation_mode == RulerPropagationMode::pull) {
          ruler = bits::set_root_flag(ruler);
        }
        succ_array[ruler_local] = ruler;
        rank_array[ruler_local] = 0;
      } else {
        node_type[ruler_local] = NodeType::ruler;
      }
      if (dist.is_local(succ, comm.rank())) {
        enqueue_locally(
            {.target_idx = succ, .ruler = ruler, .dist_from_ruler = dist_to_succ});
        continue;
      }
      send_to({.target_idx = succ, .ruler = ruler, .dist_from_ruler = dist_to_succ},
              get_succ_owner(ruler_local, succ));
      // dist.get_owner(succ));
    }
  };

  auto spawn_new_ruler = [&](auto&& enqueue_locally, auto&& send_to) {
    auto begin = std::chrono::high_resolution_clock::now();
    if (num_unreached == 0) {
      KASSERT(std::ranges::all_of(
          node_type, [](NodeType type) { return type != NodeType::unreached; }));
      return;
    }
    std::uniform_int_distribution<std::size_t> distribution(0, succ_array.size() - 1);
    std::size_t ruler_local{};
    if (!config.no_precompute_rulers) {
      next_ruler_it = std::ranges::find_if(
          next_ruler_it, local_indices_permuted.end(),
          [&](idx_t local_idx) { return node_type[local_idx] == NodeType::unreached; });
      KASSERT(next_ruler_it != local_indices_permuted.end());
      ruler_local = *next_ruler_it;
    } else {
      do {
        ruler_local = distribution(rng);
        if (node_type[ruler_local] == NodeType::unreached) {
          break;
        }
      } while (true);
    }
    auto end = std::chrono::high_resolution_clock::now();
    rulers.push_back(ruler_local);
    trace.track_spawn(end - begin);
    auto ruler = dist.get_global_idx(ruler_local, comm.rank());
    SPDLOG_TRACE("spawning new ruler {}", ruler);
    node_type[ruler_local] = NodeType::ruler;
    num_unreached--;
    auto succ = succ_array[ruler_local];
    auto dist_to_succ = rank_array[ruler_local];
    if (dist.is_local(succ, comm.rank())) {
      enqueue_locally(RulerMessage{
          .target_idx = succ, .ruler = ruler, .dist_from_ruler = dist_to_succ});
      return;
    }
    send_to(
        RulerMessage{.target_idx = succ, .ruler = ruler, .dist_from_ruler = dist_to_succ},
        get_succ_owner(ruler_local, succ));
    // dist.get_owner(succ));
  };

  auto work_on_item = [&](RulerMessage const& msg, auto&& enqueue_locally,
                          auto&& send_to) {
    const auto& [idx, ruler, dist_from_ruler] = msg;
    KASSERT(dist.get_owner(idx) == comm.rank());
    auto idx_local = dist.get_local_idx(idx, comm.rank());
    auto succ = succ_array[idx_local];
    auto dist_to_succ = rank_array[idx_local];
    succ_array[idx_local] = ruler;
    rank_array[idx_local] = dist_from_ruler;
    if (node_type[idx_local] == NodeType::ruler ||
        node_type[idx_local] == NodeType::root) {
      // we stop chasing here
      trace.track_chain_end(ruler, dist_from_ruler);
      // select new unreached vertex as ruler
      if (config.spawn) {
        spawn_new_ruler(enqueue_locally, send_to);
      }
      return;
    }
    KASSERT(node_type[idx_local] == NodeType::unreached);
    node_type[idx_local] = NodeType::reached;
    num_unreached--;
    if (dist.is_local(succ, comm.rank())) {
      enqueue_locally({.target_idx = succ,
                       .ruler = ruler,
                       .dist_from_ruler = dist_from_ruler + dist_to_succ});
      return;
    }
    send_to({.target_idx = succ,
             .ruler = ruler,
             .dist_from_ruler = dist_from_ruler + dist_to_succ},
            get_succ_owner(idx_local, succ));
    // succ_rank);
  };
  if (config.sync) {
    auto rounds = ruler_chasing_engine(config, init, work_on_item, dist, comm, grid_comm,
                                       ruler_chasing::sync);
    trace.track_ruler_chasing_rounds(rounds);
  } else {
    ruler_chasing_engine(config, init, work_on_item, dist, comm, ruler_chasing::async);
  }
  kamping::measurements::timer().stop();
  KASSERT(std::ranges::all_of(node_type,
                              [&](NodeType type) {
                                return type == NodeType::root ||
                                       type == NodeType::ruler ||
                                       type == NodeType::leaf ||
                                       type == NodeType::reached;
                              }),
          "Not all nodes were reached during ruler chasing!");

  // resetting the leafs' msb
  for (auto local_idx : rulers) {
    succ_array[local_idx] = bits::clear_root_flag(succ_array[local_idx]);
  }

  {
    kamping::measurements::timer().start("pack_base_case");
    std::vector<idx_t> succ_array_base(rulers.size());
    std::vector<rank_t> rank_array_base(rulers.size());
    auto [dist_base, unpack] =
        pack(succ_array, rank_array, dist, rulers, succ_array_base, rank_array_base, comm,
             grid_comm, config.use_grid_communication);
    kamping::measurements::timer().stop();

    kamping::measurements::timer().synchronize_and_start("base_case");
    trace.track_base_case(succ_array_base.size());
    base_algorithm(succ_array_base, rank_array_base, dist_base, comm);
    kamping::measurements::timer().stop();

    kamping::measurements::timer().start("unpack_base_case");
    unpack(succ_array_base, rank_array_base, dist_base, succ_array, rank_array, dist,
           rulers, comm, grid_comm, config.use_grid_communication);
    kamping::measurements::timer().stop();
  }

  ///////////////////////////////////////
  // Request rank and root from rulers //
  ///////////////////////////////////////
  kamping::measurements::timer().synchronize_and_start("ruler_propagation");
  ruler_propagation(config, succ_array, rank_array, inital_succ_array, node_type, rulers,
                    dist, comm, grid_comm);
  kamping::measurements::timer().stop();
  kamping::measurements::timer().synchronize_and_start("post_invert");
  if (config.post_invert) {
    post_invert(config, succ_array, rank_array, node_type, dist, comm, grid_comm);
  }
  kamping::measurements::timer().stop();
}
}  // namespace
}  // namespace sparse_ruling_set_detail

void sparse_ruling_set(SparseRulingSetConfig const& config,
                       std::span<idx_t> succ_array,
                       std::span<rank_t> rank_array,
                       Distribution const& dist,
                       kamping::Communicator<> const& comm) {
  using namespace sparse_ruling_set_detail;
  BaseAlgorithm base_algorithm;
  switch (config.base_algorithm) {
    case Algorithm::PointerDoubling:
      base_algorithm = [&](auto&&... args) {
        pointer_doubling(
            std::any_cast<PointerDoublingConfig>(config.base_algorithm_config), args...);
      };
      break;
    case Algorithm::GatherChase:
      base_algorithm = [&](auto&&... args) { kascade::rank_on_root(args...); };
      break;
    case Algorithm::AsyncPointerDoubling:
      base_algorithm = [&](auto&&... args) {
        async_pointer_doubling(
            std::any_cast<AsyncPointerChasingConfig>(config.base_algorithm_config),
            args...);
      };
      break;
    case Algorithm::RMAPointerDoubling:
      base_algorithm = [&](auto&&... args) {
        rma_pointer_doubling(
            std::any_cast<RMAPointerChasingConfig>(config.base_algorithm_config),
            args...);
      };
      break;
    case Algorithm::SparseRulingSet:
      base_algorithm = [&](auto&&... args) {
        auto nested_config =
            std::any_cast<SparseRulingSetConfig>(config.base_algorithm_config);
        // avoid infinite recursion by setting the base algorithm of the nested config to
        // a non-recursive one
        nested_config.base_algorithm = kascade::Algorithm::PointerDoubling;
        nested_config.base_algorithm_config = PointerDoublingConfig{};
        sparse_ruling_set(nested_config, args...);
      };
      break;
    default:
      throw std::runtime_error("Invalid base algorithm selected for sparse ruling set");
  }
  sparse_ruling_set(config, succ_array, rank_array, dist, base_algorithm, comm);
}
}  // namespace kascade
