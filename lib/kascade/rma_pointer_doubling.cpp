#include <algorithm>
#include <functional>
#include <ranges>
#include <string>

#include <mpi.h>

#include <fmt/ranges.h>
#include <kamping/mpi_datatype.hpp>
#include <spdlog/spdlog.h>

#include "kascade/distribution.hpp"
#include "kascade/pointer_doubling.hpp"
#include "kascade/types.hpp"

namespace kascade {

struct Entry {
  idx_t rank;
  idx_t parent;
};

void rma_pointer_doubling(std::span<const idx_t> succ_array,
                          std::span<idx_t> rank_array,
                          std::span<idx_t> root_array,
                          kamping::Communicator<> const& comm) {
  Distribution dist(succ_array.size(), comm);
  std::vector<Entry> data_array(succ_array.size());
  auto ranks =
      data_array | std::views::transform([](auto& entry) -> auto& { return entry.rank; });
  auto parents = data_array |
                 std::views::transform([](auto& entry) -> auto& { return entry.parent; });
  std::ranges::copy(succ_array, std::ranges::begin(parents));
  std::ranges::fill(ranks, 1);

  for (std::size_t local_idx = 0; local_idx < succ_array.size(); local_idx++) {
    idx_t global_idx = dist.get_global_idx(local_idx, comm.rank());
    auto& entry = data_array[local_idx];
    if (entry.parent == global_idx) {
      entry.rank = 0;
    }
  }
  MPI_Win win = MPI_WIN_NULL;
  MPI_Info info = MPI_INFO_NULL;
  MPI_Info_create(&info);
  MPI_Info_set(info, "accumulate_ops", "same_op_no_op");
  auto granularity = std::to_string(sizeof(Entry));
  MPI_Info_set(info, "mpi_accumulate_granularity", granularity.c_str());
  MPI_Info_set(info, "same_disp_unit", "true");
  MPI_Info_set(info, "mpi_assert_memory_alloc_kinds", "system");
  MPI_Win_create(data_array.data(),
                 std::ranges::ssize(data_array) * static_cast<MPI_Aint>(sizeof(Entry)),
                 sizeof(Entry), info, comm.mpi_communicator(), &win);
  MPI_Info_free(&info);
  std::vector<bool> converged(data_array.size(), false);
  std::size_t converged_indices = 0;
  while (converged_indices != converged.size()) {
    for (std::size_t i = 0; i < succ_array.size(); i++) {
      if (converged[i]) {
        continue;
      }
      MPI_Win_lock(MPI_LOCK_SHARED, comm.rank_signed(), MPI_MODE_NOCHECK, win);
      Entry entry_i{};
      MPI_Get_accumulate(nullptr, 0, MPI_DATATYPE_NULL, &entry_i, 1,
                         kamping::mpi_datatype<Entry>(), comm.rank_signed(),
                         static_cast<MPI_Aint>(i), 1, kamping::mpi_datatype<Entry>(),
                         MPI_NO_OP, win);
      MPI_Win_unlock(comm.rank_signed(), win);

      auto parent_rank = dist.get_owner(entry_i.parent);
      auto parent_offset = dist.get_local_idx(entry_i.parent, parent_rank);
      MPI_Win_lock(MPI_LOCK_SHARED, static_cast<int>(parent_rank), MPI_MODE_NOCHECK, win);
      Entry parent_entry{};
      MPI_Get_accumulate(nullptr, 0, MPI_DATATYPE_NULL, &parent_entry, 1,
                         kamping::mpi_datatype<Entry>(), static_cast<int>(parent_rank),
                         static_cast<MPI_Aint>(parent_offset), 1,
                         kamping::mpi_datatype<Entry>(), MPI_NO_OP, win);
      MPI_Win_unlock(static_cast<int>(parent_rank), win);
      if (parent_entry.rank == 0) {
        converged[i] = true;
        converged_indices++;
        continue;
      }

      entry_i.rank += parent_entry.rank;
      entry_i.parent = parent_entry.parent;
      MPI_Win_lock(MPI_LOCK_SHARED, comm.rank_signed(), MPI_MODE_NOCHECK, win);
      MPI_Accumulate(&entry_i, 1, kamping::mpi_datatype<Entry>(), comm.rank_signed(),
                     static_cast<MPI_Aint>(i), 1, kamping::mpi_datatype<Entry>(),
                     MPI_REPLACE, win);
      MPI_Win_unlock(comm.rank_signed(), win);
    }
  }

  MPI_Win_free(&win);
  std::ranges::copy(ranks, rank_array.begin());
  std::ranges::copy(parents, root_array.begin());
}
}  // namespace kascade
