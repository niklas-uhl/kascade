#include "generation.hpp"

#include <kagen.h>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/exscan.hpp>
#include <kamping/types/unsafe/tuple.hpp>
#include <kamping/types/unsafe/utility.hpp>
#include <kassert/kassert.hpp>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include "kascade/eulertour.hpp"

namespace {
// Taken from the KaCCv2 library by Tim Niklas Uhl
template <typename T, typename NodeIDT>
auto reshape(std::vector<T> const& v,
             std::vector<NodeIDT> dist,
             kamping::Communicator<> const& comm) {
  namespace kmp = kamping::params;
  // compute my local range of v
  std::pair<NodeIDT, NodeIDT> local_range{};
  local_range.first = v.size();
  comm.exscan_inplace(kmp::send_recv_buf(local_range.first), kmp::op(std::plus<>{}),
                      kmp::values_on_rank_0(0));
  local_range.second = local_range.first + v.size();

  MPI_Win win = MPI_WIN_NULL;
  auto new_local_size = dist[comm.rank() + 1] - dist[comm.rank()];
  std::vector<T> v_new(new_local_size);

  // window setup
  MPI_Info info = MPI_INFO_NULL;
  MPI_Info_create(&info);
  MPI_Info_set(info, "no_locks", "true");
  MPI_Info_set(info, "same_disp_unit", "true");
  MPI_Win_create(v_new.data(), v_new.size() * sizeof(T), sizeof(T), info,
                 comm.mpi_communicator(), &win);
  MPI_Info_free(&info);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
  for (std::size_t rank = 0; rank < comm.size(); ++rank) {
    // compute the local range of v for this rank
    auto rank_local_range = std::pair{dist[rank], dist[rank + 1]};
    auto overlap = std::pair{std::max(local_range.first, rank_local_range.first),
                             std::min(local_range.second, rank_local_range.second)};
    if (overlap.first >= overlap.second) {
      continue;  // no overlap, skip
    }
    std::span<const T> chunk = std::span(v).subspan(overlap.first - local_range.first,
                                                    overlap.second - overlap.first);
    MPI_Put(chunk.data(), chunk.size(), kamping::mpi_datatype<T>(), rank,
            overlap.first - rank_local_range.first, chunk.size(),
            kamping::mpi_datatype<T>(), win);
  }
  // end epoch
  MPI_Win_fence(MPI_MODE_NOSUCCEED | MPI_MODE_NOSTORE, win);
  MPI_Win_free(&win);
  return v_new;
}
}  // namespace

namespace kascade::input {
auto generate_input(Config const& config, kamping::Communicator<> const& comm)
    -> std::vector<kascade::idx_t> {
  kagen::KaGen gen(comm.mpi_communicator());
  gen.EnableBasicStatistics();
  gen.UseCSRRepresentation();
  switch (config.input_processing) {
    case InputProcessing::none: {
      spdlog::stopwatch stopwatch;
      auto G = gen.GenerateFromOptionString(config.kagen_option_string);
      auto succ_array = internal::write_graph_to_succ_array(G, comm);
      spdlog::get("root")->info("Generation finished in {} seconds.", stopwatch);
      return succ_array;
    }
    case InputProcessing::bfs: {
      auto G = gen.GenerateFromOptionString(config.kagen_option_string);
      auto succ_array = internal::generate_bfs_tree(G, comm);
      return succ_array;
    }
    case InputProcessing::eulertour:
    case InputProcessing::eulertour_break_high_degree: {
      auto G = gen.GenerateFromOptionString(config.kagen_option_string);
      auto parent_array = internal::generate_bfs_tree(G, comm);
      auto succ_array = compute_euler_tour(
          parent_array, comm,
          config.input_processing == InputProcessing::eulertour_break_high_degree);
      return succ_array;
    }
    case InputProcessing::invalid:
      throw std::runtime_error("Invalid input processing selected.");
    default:
      return {};
  }
}

auto rebalance_input(std::vector<idx_t> succ_array, kamping::Communicator<> const& comm)
    -> std::vector<kascade::idx_t> {
  namespace kmp = kamping::params;
  std::size_t total_size =
      comm.allreduce_single(kmp::send_buf(succ_array.size()), kmp::op(std::plus<>{}));
  std::size_t local_chunk_size = total_size / comm.size();
  std::vector<std::size_t> distribution(comm.size() + 1);
  std::fill(distribution.begin(), distribution.begin() + comm.size_signed(),
            local_chunk_size);
  for (std::size_t i = 0; i < (total_size % comm.size()); ++i) {
    distribution[i]++;
  }
  std::exclusive_scan(distribution.begin(), distribution.end(), distribution.begin(),
                      std::size_t{0});
  SPDLOG_LOGGER_TRACE(spdlog::get("root"), "total size {} dist {}", total_size,
                      distribution);
  return reshape(succ_array, distribution, comm);
}

auto get_input_distribution_stats(std::span<idx_t> succ_array,
                                  kamping::Communicator<> const& comm)
    -> InputDistributionStats {
  namespace kmp = kamping::params;
  std::size_t min_size = comm.allreduce_single(kmp::send_buf(succ_array.size()),
                                               kmp::op(kamping::ops::min<>{}));

  std::size_t max_size = comm.allreduce_single(kmp::send_buf(succ_array.size()),
                                               kmp::op(kamping::ops::max<>{}));

  std::size_t total_size =
      comm.allreduce_single(kmp::send_buf(succ_array.size()), kmp::op(std::plus<>{}));
  return InputDistributionStats{
      .min_local_size = min_size, .max_local_size = max_size, .total_size = total_size};
}

}  // namespace kascade::input
