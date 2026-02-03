#pragma once

#include <kamping/communicator.hpp>
#include <spdlog/spdlog.h>

namespace kascade {

class TopologyAwareGridCommunicator {
public:
  using GlobalCommunicator = kamping::Communicator<>;
  TopologyAwareGridCommunicator(kamping::Communicator<> const& global_comm);
  TopologyAwareGridCommunicator(kamping::Communicator<> const& global_comm,
                                std::size_t intra_comm_size);

  [[nodiscard]] auto global_comm() const -> kamping::Communicator<> const&;
  [[nodiscard]] auto inter_node_comm() const -> kamping::Communicator<> const&;
  [[nodiscard]] auto intra_node_comm() const -> kamping::Communicator<> const&;

  [[nodiscard]] auto inter_node_rank(std::size_t global_rank) const -> std::size_t;
  [[nodiscard]] auto intra_node_rank(std::size_t global_rank) const -> std::size_t;

  [[nodiscard]] auto inter_node_rank() const -> std::size_t;
  [[nodiscard]] auto intra_node_rank() const -> std::size_t;
  [[nodiscard]] auto last_intra_node_comm_size() const -> std::size_t;
  [[nodiscard]] auto is_complete() const -> bool;
  [[nodiscard]] auto ranks_per_compute_node() const -> std::size_t;
  [[nodiscard]] auto in_last_intra_node_communicator(std::size_t global_rank) const
      -> bool;

private:
  GlobalCommunicator const* global_comm_;
  std::size_t ranks_per_compute_node_;
  std::size_t inter_node_comm_size_;
  kamping::Communicator<> inter_node_comm_;
  kamping::Communicator<> intra_node_comm_;
};

class TopologyAwareGridRouting {
public:
  TopologyAwareGridRouting(TopologyAwareGridCommunicator const& grid_comm)
      : grid_comm_{&grid_comm} {}
  auto is_next_hop_direct(std::size_t sender, std::size_t receiver) -> bool;

private:
  TopologyAwareGridCommunicator const* grid_comm_;
};
}  // namespace kascade
