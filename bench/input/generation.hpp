#pragma once

#include <kagen.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/communicator.hpp>

#include "input/bfs.hpp"
#include "kascade/graph/graph.hpp"
#include "kascade/types.hpp"

namespace kascade::input::internal {
auto generate_bfs_tree(kagen::Graph const& kagen_graph,
                       kamping::Communicator<> const& comm)
    -> std::vector<kascade::idx_t>;

auto write_graph_to_succ_array(kagen::Graph const& kagen_graph,
                               kamping::Communicator<> const& comm)
    -> std::vector<kascade::idx_t>;
}  // namespace kascade::input::internal

namespace kascade::input {
enum class InputProcessing : std::uint8_t {
  none,
  bfs,
  eulertour,
  invalid,
};
struct Config {
  std::string kagen_option_string;
  InputProcessing input_processing;
};

struct InputDistributionStats {
  std::size_t min_local_size;
  std::size_t max_local_size;
  std::size_t total_size;
};

auto generate_input(Config const& config, kamping::Communicator<> const& comm)
    -> std::vector<kascade::idx_t>;

auto rebalance_input(std::vector<idx_t> succ_array, kamping::Communicator<> const& comm)
    -> std::vector<kascade::idx_t>;

auto get_input_distribution_stats(std::span<idx_t> succ_array,
                                  kamping::Communicator<> const& comm)
    -> InputDistributionStats;

}  // namespace kascade::input
