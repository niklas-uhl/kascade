#include "mplr.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

#include "detail/mplr/forest_regular_optimized_ruling_set.hpp"

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
}  // namespace mplr
