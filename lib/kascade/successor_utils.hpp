#pragma once

#include <ranges>
#include <span>
#include <vector>

#include <kamping/collectives/gather.hpp>
#include <kamping/communicator.hpp>
#include <spdlog/spdlog.h>

#include "grid_communicator.hpp"
#include "kascade/distribution.hpp"
#include "kascade/eulertour.hpp"
#include "kascade/graph/graph.hpp"
#include "kascade/types.hpp"

namespace kascade {

auto is_list(std::span<const idx_t> succ_array, kamping::Communicator<> const& comm)
    -> bool;

auto is_list(std::span<const idx_t> succ_array,
             Distribution const& dist,
             kamping::Communicator<> const& comm) -> bool;

auto reverse_list(std::span<const idx_t> succ_array,
                  std::span<const rank_t> dist_to_succ,
                  std::span<idx_t> pred_array,
                  std::span<rank_t> dist_to_pred,
                  Distribution const& dist,
                  kamping::Communicator<> const& comm,
                  bool use_grid_comm = false) -> std::vector<idx_t>;
  
auto reverse_list(std::span<const idx_t> succ_array,
                  std::span<const rank_t> dist_to_succ,
                  std::span<idx_t> pred_array,
                  std::span<rank_t> dist_to_pred,
                  Distribution const& dist,
                  kamping::Communicator<> const& comm,
                  std::optional<TopologyAwareGridCommunicator> const& grid_comm,
                  bool use_grid_comm = false) -> std::vector<idx_t>;

template <typename T>
class maybe_owning_contiguous_range {
public:
  maybe_owning_contiguous_range(std::span<T const> span) noexcept : span_(span) {}
  maybe_owning_contiguous_range(std::vector<T> v) noexcept
      : storage_(std::move(v)), span_(*storage_) {}

  [[nodiscard]] auto span() const noexcept -> std::span<const T> { return span_; }
  [[nodiscard]] auto owns_memory() const noexcept -> bool { return storage_.has_value(); }

private:
  std::optional<std::vector<T>> storage_;
  std::span<T const> span_;
};

struct ReversedTree {
  std::size_t num_proxy_vertices{};
  maybe_owning_contiguous_range<idx_t> parent_array;
  graph::DistributedCSRGraph tree;
};
auto reverse_rooted_tree(std::span<const idx_t> succ_array,
                         std::span<const rank_t> dist_to_succ,
                         Distribution const& dist,
                         kamping::Communicator<> const& comm,
                         bool add_back_edge) -> ReversedTree;

struct resolve_high_degree_tag {};
static constexpr resolve_high_degree_tag resolve_high_degree{};

auto reverse_rooted_tree(std::span<const idx_t> succ_array,
                         std::span<const rank_t> dist_to_succ,
                         Distribution const& dist,
                         kamping::Communicator<> const& comm,
                         bool add_back_edge,
                         resolve_high_degree_tag) -> ReversedTree;

auto is_root(std::size_t local_idx,
             std::span<const idx_t> succ_array,
             Distribution const& dist,
             kamping::Communicator<> const& comm) -> bool;

struct LeafInfo {
private:
  std::vector<bool> has_pred_;  // per-local index
  Distribution const* dist_;
  kamping::Communicator<> const* comm_;
  std::size_t num_local_leaves_;

public:
  LeafInfo(std::span<const idx_t> succ_array,
           Distribution const& dist,
           kamping::Communicator<> const& comm);

  [[nodiscard]] auto is_leaf(idx_t local_idx) const -> bool;

  [[nodiscard]] auto num_local_leaves() const -> std::size_t;
  [[nodiscard]] auto leaves() const {
    auto indices = std::views::iota(idx_t{0}, static_cast<idx_t>(has_pred_.size()));

    return indices |
           std::views::filter([&](auto local_idx) { return this->is_leaf(local_idx); });
  }
};

auto roots(std::span<const idx_t> succ_array,
           Distribution const& dist,
           kamping::Communicator<> const& comm) -> std::vector<idx_t>;

auto leaves(std::span<const idx_t> succ_array,
            Distribution const& dist,
            kamping::Communicator<> const& comm) -> std::vector<idx_t>;

auto trace_successor_list(std::span<const idx_t> root_array,
                          std::span<const rank_t> rank_array,
                          const kamping::Communicator<>& comm) -> std::string;
}  // namespace kascade
