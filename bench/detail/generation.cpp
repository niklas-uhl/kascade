#include "detail/generation.hpp"

#include <numeric>

#include <kagen.h>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/exscan.hpp>
#include <kamping/types/unsafe/tuple.hpp>
#include <kamping/types/unsafe/utility.hpp>
#include <kassert/kassert.hpp>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include "detail/input_distribution_utils.hpp"

namespace {

auto path_to_succ_array(kascade::idx_t size,
                        kagen::Edgelist path,
                        kamping::Communicator<> const& comm)
    -> std::vector<kascade::idx_t> {
  namespace kmp = kamping::params;

  kascade::bench::Distribution dist(size, comm.size());
  std::size_t prefix = dist.get_prefix(comm.rank());
  std::vector<kascade::idx_t> succ_array(dist.get_count(comm.rank()));
  std::ranges::iota(succ_array, prefix);
  std::ranges::sort(path);
  std::vector<int> send_counts(comm.size(), 0);

  // could also do some kind of binary search
  for (auto const& [u, v] : path) {
    auto owner = dist.get_owner(u);
    ++send_counts[owner];
  }

  auto recv_buf = comm.alltoallv(kmp::send_buf(path), kmp::send_counts(send_counts));

  for (auto const& [cur, next] : recv_buf) {
    KASSERT(cur >= prefix);
    succ_array[cur - prefix] = static_cast<kascade::idx_t>(next);
  }
  return succ_array;
}
}  // namespace

auto generate_input(Config const& config, kamping::Communicator<> const& comm)
    -> std::vector<kascade::idx_t> {
  kagen::KaGen gen(comm.mpi_communicator());
  gen.EnableBasicStatistics();
  gen.UseEdgeListRepresentation();
  spdlog::stopwatch stopwatch;
  // TODO do proper dispatch
  auto G = gen.GenerateFromOptionString(config.kagen_option_string);
  auto succ_array = path_to_succ_array(G.NumberOfGlobalVertices(), G.edges, comm);
  spdlog::get("root")->info("Generation finished in {} seconds.", stopwatch);
  return succ_array;
}
