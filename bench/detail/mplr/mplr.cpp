#include "mplr.hpp"
#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

#include "detail/mplr/forest_regular_optimized_ruling_set.hpp"
#include "kascade/distribution.hpp"

#pragma GCC diagnostic pop

namespace mplr {
auto forest_ruling_set(Configuration config,
                       std::vector<std::uint64_t>& succ_array,
                       kamping::Communicator<>& comm)
    -> std::pair<std::vector<std::uint64_t>, std::vector<std::int64_t>> {
  forest_regular_optimized_ruling_set algorithm{
      config.comm_rounds, static_cast<std::uint32_t>(config.recursion_levels),
      config.use_grid, config.use_aggregation};
  karam::mpi::GridCommunicator grid_comm(comm);
  algorithm.start(succ_array, comm, grid_comm);
  return std::make_pair(std::move(algorithm.result_root),
                        std::move(algorithm.result_dist));
}

auto forest_pointer_doubling(Configuration config,
                             std::vector<std::uint64_t>& succ_array,
                             kamping::Communicator<>& comm)
    -> std::pair<std::vector<std::uint64_t>, std::vector<std::int64_t>> {
  kascade::Distribution dist(succ_array.size(), comm);
  std::vector<std::int64_t> rank_array(succ_array.size(), 1);
  std::vector<std::uint32_t> target_pe(succ_array.size(), 1);
  std::vector<std::uint64_t> global_index(succ_array.size(), 1);
  for (std::size_t i = 0; i < succ_array.size(); ++i) {
    global_index[i] = dist.get_global_idx(i, comm.rank());
    target_pe[i] = dist.get_owner(succ_array[i]);
    if (succ_array[i] == dist.get_global_idx(i, comm.rank())) {
      rank_array[i] = 0;
    }
  }

  std::uint64_t offset = std::accumulate(
      dist.counts().begin(), dist.counts().begin() + comm.rank_signed(), std::size_t{0});

  forest_irregular_pointer_doubling algorithm{succ_array,
                                              rank_array,
                                              target_pe,
                                              offset,
                                              dist.get_global_size(),
                                              global_index,
                                              config.use_grid,
                                              config.use_aggregation};
  karam::mpi::GridCommunicator grid_comm(comm);
  algorithm.start(comm, grid_comm);
  return std::make_pair(std::move(algorithm.local_rulers), std::move(algorithm.r));
}
}  // namespace mplr
