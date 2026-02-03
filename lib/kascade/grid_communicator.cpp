#include "kascade/grid_communicator.hpp"

#include <numeric>
#include <ranges>

#include <kamping/collectives/allreduce.hpp>

namespace {
template <class T>
constexpr auto ceil_div(T num, T den) -> T {
  return (num + den - 1) / den;
}

}  // namespace

namespace kascade {

// assumes consecutive world ranks per compute node
TopologyAwareGridCommunicator::TopologyAwareGridCommunicator(
    kamping::Communicator<> const& global_comm)
    : global_comm_{&global_comm} {
  namespace kmp = kamping::params;

  // create intra-node communicator
  intra_node_comm_ = global_comm_->split_to_shared_memory();
  ranks_per_compute_node_ = global_comm_->allreduce_single(
      kmp::send_buf(intra_node_comm_.size()), kmp::op(kamping::ops::max<>{}));
  inter_node_comm_size_ = ceil_div(global_comm.size(), ranks_per_compute_node_);

  // create inter-node communicator
  int rank_within_inter_node_comm =
      static_cast<int>(global_comm.rank() % ranks_per_compute_node_);
  auto inter_node_group =
      std::ranges::views::iota(rank_within_inter_node_comm, global_comm.size_signed()) |
      std::views::stride(static_cast<int>(ranks_per_compute_node_)) |
      std::ranges::to<std::vector<int>>();
  inter_node_comm_ = global_comm_->create_subcommunicators(inter_node_group);
  spdlog::get("gather")->info("inter-comm-size: {}, intra-comm-size {}",
                              inter_node_comm_.size(), intra_node_comm_.size());
}

TopologyAwareGridCommunicator::TopologyAwareGridCommunicator(
    kamping::Communicator<> const& global_comm,
    std::size_t intra_comm_size)
    : global_comm_{&global_comm},
      ranks_per_compute_node_{intra_comm_size},
      inter_node_comm_size_{ceil_div(global_comm.size(), ranks_per_compute_node_)} {
  if (intra_comm_size == 1) {
    // minimize initialization costs in this case
    intra_node_comm_ = kamping::Communicator(MPI_COMM_SELF);
    inter_node_comm_ = kamping::Communicator(global_comm.mpi_communicator());
    return;
  }
  // create intra-node communicator
  intra_node_comm_ =
      global_comm_->split(static_cast<int>(global_comm_->rank() / intra_comm_size));

  // create inter-node communicator
  int rank_within_inter_node_comm =
      static_cast<int>(global_comm.rank() % ranks_per_compute_node_);
  auto inter_node_group =
      std::ranges::views::iota(rank_within_inter_node_comm, global_comm.size_signed()) |
      std::views::stride(static_cast<int>(ranks_per_compute_node_)) |
      std::ranges::to<std::vector<int>>();
  inter_node_comm_ = global_comm_->create_subcommunicators(inter_node_group);
  spdlog::get("gather")->info("inter-comm-size: {}, intra-comm-size {}",
                              inter_node_comm_.size(), intra_node_comm_.size());
}
auto TopologyAwareGridCommunicator::global_comm() const
    -> kamping::Communicator<> const& {
  return *global_comm_;
}

auto TopologyAwareGridCommunicator::inter_node_comm() const
    -> kamping::Communicator<> const& {
  return inter_node_comm_;
}

auto TopologyAwareGridCommunicator::intra_node_comm() const
    -> kamping::Communicator<> const& {
  return intra_node_comm_;
}
auto TopologyAwareGridCommunicator::inter_node_rank(std::size_t global_rank) const
    -> std::size_t {
  return global_rank / ranks_per_compute_node_;
}

auto TopologyAwareGridCommunicator::intra_node_rank(std::size_t global_rank) const
    -> std::size_t {
  return global_rank % ranks_per_compute_node_;
}
auto TopologyAwareGridCommunicator::inter_node_rank() const -> std::size_t {
  return inter_node_comm_.rank();
}

auto TopologyAwareGridCommunicator::intra_node_rank() const -> std::size_t {
  return intra_node_comm_.rank();
}

auto TopologyAwareGridCommunicator::last_intra_node_comm_size() const -> std::size_t {
  return global_comm_->size() % ranks_per_compute_node_;
}

auto TopologyAwareGridCommunicator::is_complete() const -> bool {
  return global_comm_->size() % ranks_per_compute_node_ == 0;
}

auto TopologyAwareGridCommunicator::ranks_per_compute_node() const -> std::size_t {
  return ranks_per_compute_node_;
}

auto TopologyAwareGridCommunicator::in_last_intra_node_communicator(
    std::size_t global_rank) const -> bool {
  return inter_node_rank(global_rank) + 1 == inter_node_comm_size_;
}

auto TopologyAwareGridRouting::is_next_hop_direct(std::size_t sender,
                                                  std::size_t receiver) -> bool {
  if (sender >= receiver) {
    // only last intra node communicator may be incomplete
    return true;
  }
  std::int64_t const inter_node_distance =
      static_cast<std::int64_t>(grid_comm_->inter_node_rank(receiver)) -
      static_cast<std::int64_t>(grid_comm_->inter_node_rank(sender));
  auto receiver_proxy = static_cast<std::size_t>(
      static_cast<int64_t>(sender) +
      (inter_node_distance * grid_comm_->ranks_per_compute_node()));
  return receiver_proxy < grid_comm_->global_comm().size();
}
}  // namespace kascade
