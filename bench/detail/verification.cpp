#include "detail/verification.hpp"

#include <ranges>

#include <kamping/collectives/allreduce.hpp>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

auto verify(std::span<const kascade::idx_t> /* succ */,
            AbstractAlgorithm& reference,
            AbstractAlgorithm& algorithm,
            std::size_t verify_level,
            bool continue_on_mismatch,
            const kamping::Communicator<>& comm) -> void {
  if (verify_level == 0) {
    return;
  }
  auto const& ref_root = reference.get_root_array();
  auto const& root = algorithm.get_root_array();

  if (root.size() != ref_root.size()) {
    SPDLOG_ERROR("Root array size does not match reference: {} != {}", root.size(),
                 ref_root.size());
    if (!continue_on_mismatch) {
      std::exit(1);
    }
  }
  bool mismatch_found = false;
  for (auto [idx, roots] : std::views::zip(root, ref_root) | std::views::enumerate) {
    auto& [r, r_ref] = roots;
    if (r != r_ref) {
      mismatch_found = true;
      SPDLOG_ERROR("Root array does not match reference: {} != {} at index {}", r, r_ref,
                   idx);
      if (!continue_on_mismatch) {
        std::exit(1);
      }
      break;
    }
  }
  comm.allreduce(kamping::send_recv_buf(mismatch_found),
                 kamping::op(std::logical_or<>{}));
  if (mismatch_found) {
    spdlog::trace("roots={}, reference={}", root, ref_root);
    SPDLOG_LOGGER_ERROR(spdlog::get("root"), "Root array does not match reference.");
    if (!continue_on_mismatch) {
      std::exit(1);
    }
  } else {
    SPDLOG_LOGGER_INFO(spdlog::get("root"), "Root array matches reference.");
  }

  auto const& ref_rank = reference.get_rank_array();
  auto const& rank = algorithm.get_rank_array();
  mismatch_found = false;
  if (rank.size() != ref_rank.size()) {
    SPDLOG_LOGGER_ERROR(spdlog::get("root"),
                        "Rank array size does not match reference: {} != {}", rank.size(),
                        ref_rank.size());
    if (!continue_on_mismatch) {
      std::exit(1);
    }
  }
  for (auto [idx, ranks] : std::views::zip(rank, ref_rank) | std::views::enumerate) {
    auto& [rk, rk_ref] = ranks;
    if (rk != rk_ref) {
      mismatch_found = true;
      SPDLOG_LOGGER_ERROR(spdlog::get("root"),
                          "Rank array does not match reference: {} != {} at index {}", rk,
                          rk_ref, idx);
      if (!continue_on_mismatch) {
        std::exit(1);
      }
      break;
    }
  }
  comm.allreduce(kamping::send_recv_buf(mismatch_found),
                 kamping::op(std::logical_or<>{}));
  if (mismatch_found) {
    spdlog::trace("ranks={}, reference={}", rank, ref_rank);
    SPDLOG_LOGGER_ERROR(spdlog::get("root"), "Rank array does not match reference.");
    if (!continue_on_mismatch) {
      std::exit(1);
    }
  } else {
    SPDLOG_LOGGER_INFO(spdlog::get("root"), "Rank array matches reference .");
  }
}
