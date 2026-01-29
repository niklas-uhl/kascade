#include <algorithm>
#include <functional>
#include <ranges>
#include <string>

#include <mpi.h>

#include <fmt/ranges.h>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/mpi_datatype.hpp>
#include <spdlog/spdlog.h>

#include "kascade/configuration.hpp"
#include "kascade/distribution.hpp"
#include "kascade/pointer_doubling.hpp"
#include "kascade/types.hpp"

namespace kascade {

struct Entry {
  idx_t rank;
  idx_t parent;
};

namespace rma::sync_mode {
struct passive_target_t {};
struct fenced_t {};
static constexpr passive_target_t passive_target{};
static constexpr fenced_t fenced{};
}  // namespace rma::sync_mode

namespace {
void rma_pointer_doubling(RMAPointerChasingConfig const& config,
                          std::span<idx_t> succ_array,
                          std::span<idx_t> rank_array,
                          Distribution const& dist,
                          kamping::Communicator<> const& comm,
                          rma::sync_mode::passive_target_t /* tag */) {
  std::vector<Entry> data_array(succ_array.size());
  auto ranks =
      data_array | std::views::transform([](auto& entry) -> auto& { return entry.rank; });
  auto parents = data_array |
                 std::views::transform([](auto& entry) -> auto& { return entry.parent; });
  std::ranges::copy(succ_array, std::ranges::begin(parents));
  std::ranges::copy(rank_array, std::ranges::begin(ranks));

  MPI_Win win = MPI_WIN_NULL;
  MPI_Info info = MPI_INFO_NULL;
  MPI_Info_create(&info);
  MPI_Info_set(info, "accumulate_ops", "same_op_no_op");
  auto granularity = std::to_string(sizeof(Entry));
  // according to MPI standard, accumulate_ordering only refers to operations from the
  // same rank to the same target location. Since we only access one location per epoch,
  // we can set it to none (hopefully improving performance)
  MPI_Info_set(info, "accumulate_ordering", "none");
  MPI_Info_set(info, "mpi_accumulate_granularity", granularity.c_str());
  MPI_Info_set(info, "same_disp_unit", "true");
  MPI_Info_set(info, "mpi_assert_memory_alloc_kinds", "system");
  MPI_Win_create(data_array.data(),
                 std::ranges::ssize(data_array) * static_cast<MPI_Aint>(sizeof(Entry)),
                 sizeof(Entry), info, comm.mpi_communicator(), &win);
  MPI_Info_free(&info);
  std::vector<bool> converged(data_array.size(), false);
  std::size_t converged_indices = 0;
  for (std::size_t local_idx = 0; local_idx < data_array.size(); local_idx++) {
    if (data_array[local_idx].parent == dist.get_global_idx(local_idx, comm.rank())) {
      KASSERT(data_array[local_idx].rank == 0);
      converged[local_idx] = true;
      converged_indices++;
    }
  }

  auto batch_size = std::min(config.batch_size, data_array.size());
  SPDLOG_LOGGER_INFO(spdlog::get("root"),
                     "passive target RMA pointer doubling, batch_size={}", batch_size);
  auto indices = std::views::iota(std::size_t{0}, data_array.size());
  auto batches = indices | std::views::chunk(batch_size);
  std::vector<Entry> buffer(batch_size);
  while (converged_indices != converged.size()) {
    SPDLOG_TRACE("converged={}/{}", converged_indices, converged.size());
    for (auto batch : batches) {
      // fetch parent entries
      MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
      for (auto [idx, buffer] : std::views::zip(batch, buffer)) {
        if (converged[idx]) {
          continue;
        }
        auto& entry_i = data_array[idx];
        auto parent_rank = dist.get_owner(entry_i.parent);
        auto parent_offset = dist.get_local_idx(entry_i.parent, parent_rank);
        if (parent_rank == comm.rank()) {
          // we can read locally via load, since nobody we only update our local array by
          // ourselves
          buffer = data_array[parent_offset];
          continue;
        }
        // this get might happen concurrently with a local store on the target rank, so we
        // have to use an atomic operation
        MPI_Get_accumulate(nullptr, 0, MPI_DATATYPE_NULL, &buffer, 1,
                           kamping::mpi_datatype<Entry>(), static_cast<int>(parent_rank),
                           static_cast<MPI_Aint>(parent_offset), 1,
                           kamping::mpi_datatype<Entry>(), MPI_NO_OP, win);
      }
      MPI_Win_unlock_all(win);
      // update local entries
      for (auto [idx, buffer_entry] : std::views::zip(batch, buffer)) {
        if (converged[idx]) {
          // already converged, copy the old value back
          buffer_entry = data_array[idx];
          continue;
        }
        if (buffer_entry.rank == 0) {
          converged[idx] = true;
          converged_indices++;
          // copy the old value back
          buffer_entry = data_array[idx];
          continue;
        }
        buffer_entry.rank += data_array[idx].rank;
        // buffer_entry.parent = buffer_entry.parent // NOOP, already set
      }
      // write back updated entries batched
      MPI_Win_lock(MPI_LOCK_SHARED, comm.rank_signed(), MPI_MODE_NOCHECK, win);
      MPI_Accumulate(buffer.data(), static_cast<int>(std::ranges::size(batch)),
                     kamping::mpi_datatype<Entry>(), comm.rank_signed(),
                     static_cast<MPI_Aint>(batch.front()),
                     static_cast<int>(std::ranges::size(batch)),
                     kamping::mpi_datatype<Entry>(), MPI_REPLACE, win);
      MPI_Win_unlock(comm.rank_signed(), win);
    }
  }

  MPI_Win_free(&win);
  std::ranges::copy(ranks, rank_array.begin());
  std::ranges::copy(parents, succ_array.begin());
}

void rma_pointer_doubling(RMAPointerChasingConfig const& /* config */,
                          std::span<idx_t> succ_array,
                          std::span<idx_t> rank_array,
                          Distribution const& dist,
                          kamping::Communicator<> const& comm,
                          rma::sync_mode::fenced_t /* tag */) {
  std::vector<Entry> data_array(succ_array.size());
  auto ranks =
      data_array | std::views::transform([](auto& entry) -> auto& { return entry.rank; });
  auto parents = data_array |
                 std::views::transform([](auto& entry) -> auto& { return entry.parent; });
  std::ranges::copy(succ_array, std::ranges::begin(parents));
  std::ranges::copy(rank_array, std::ranges::begin(ranks));

  MPI_Win win = MPI_WIN_NULL;
  MPI_Info info = MPI_INFO_NULL;
  MPI_Info_create(&info);
  MPI_Info_set(info, "same_disp_unit", "true");
  MPI_Info_set(info, "mpi_assert_memory_alloc_kinds", "system");
  MPI_Info_set(info, "no_lock", "true");
  MPI_Win_create(data_array.data(),
                 std::ranges::ssize(data_array) * static_cast<MPI_Aint>(sizeof(Entry)),
                 sizeof(Entry), info, comm.mpi_communicator(), &win);
  MPI_Info_free(&info);
  std::vector<bool> has_converged(succ_array.size(), false);
  std::size_t converged_indices = 0;
  for (std::size_t local_idx = 0; local_idx < succ_array.size(); local_idx++) {
    if (data_array[local_idx].parent == dist.get_global_idx(local_idx, comm.rank())) {
      KASSERT(data_array[local_idx].rank == 0);
      has_converged[local_idx] = true;
      converged_indices++;
    }
  }

  std::vector<Entry> buffer(succ_array.size());
  while (true) {
    bool has_locally_converged = converged_indices == has_converged.size();
    bool has_globally_converged = has_locally_converged;
    comm.allreduce(kamping::send_recv_buf(has_globally_converged),
                   kamping::op(std::logical_and<>{}));
    if (has_globally_converged) {
      break;
    }
    SPDLOG_TRACE("converged={}/{}", converged_indices, succ_array.size());
    MPI_Win_fence(MPI_MODE_NOPRECEDE | MPI_MODE_NOPUT, win);
    // remote access epoch
    for (std::size_t i = 0; i < succ_array.size(); i++) {
      if (has_locally_converged) {
        break;
      }
      if (has_converged[i]) {
        continue;
      }
      auto& entry_i = data_array[i];
      auto parent_rank = dist.get_owner(entry_i.parent);
      auto parent_offset = dist.get_local_idx(entry_i.parent, parent_rank);
      if (parent_rank == comm.rank()) {
        buffer[i] = data_array[parent_offset];
      } else {
        MPI_Get(&buffer[i], 1, kamping::mpi_datatype<Entry>(),
                static_cast<int>(parent_rank), static_cast<MPI_Aint>(parent_offset), 1,
                kamping::mpi_datatype<Entry>(), win);
      }
    }
    MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, win);
    // local epoch
    for (std::size_t i = 0; i < succ_array.size(); i++) {
      if (has_locally_converged) {
        break;
      }
      if (has_converged[i]) {
        continue;
      }
      if (buffer[i].rank == 0) {
        has_converged[i] = true;
        converged_indices++;
      }
      auto& entry_i = data_array[i];
      entry_i.rank += buffer[i].rank;
      entry_i.parent = buffer[i].parent;
    }
  }
  MPI_Win_fence(MPI_MODE_NOPRECEDE | MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, win);

  MPI_Win_free(&win);
  std::ranges::copy(ranks, rank_array.begin());
  std::ranges::copy(parents, succ_array.begin());
}
}  // namespace

void rma_pointer_doubling(RMAPointerChasingConfig const& config,
                          std::span<idx_t> succ_array,
                          std::span<idx_t> rank_array,
                          Distribution const& dist,
                          kamping::Communicator<> const& comm) {
  switch (config.sync_mode) {
    case RMASyncMode::fenced:
      rma_pointer_doubling(config, succ_array, rank_array, dist, comm,
                           rma::sync_mode::fenced);
      break;
    case RMASyncMode::passive_target:
      rma_pointer_doubling(config, succ_array, rank_array, dist, comm,
                           rma::sync_mode::passive_target);
      break;
    case RMASyncMode::invalid:
      throw std::runtime_error("Invalid RMA sync mode");
  }
};
}  // namespace kascade
