#include "generation.hpp"

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
      spdlog::get("gather")->info("graph gen {} seconds.", succ_array);
      return succ_array;
    }
    case InputProcessing::invalid:
      throw std::runtime_error("Invalid input processing selected.");
    default:
      return {};
  }
}
}  // namespace kascade::input
