#include <kascade/list_ranking.hpp>

#include <kamping/collectives/gather.hpp>
#include <kamping/collectives/scatter.hpp>
#include <kamping/named_parameters.hpp>

#include "kascade/types.hpp"

namespace kascade {
void rank(std::span<const idx_t> succ,
          std::span<idx_t> dist,
          kamping::Communicator<> const& comm) {}

void rank_on_root(std::span<const idx_t> succ,
                  std::span<idx_t> dist,
                  kamping::Communicator<> const& comm) {
  namespace kmp = kamping::params;
  auto [succ_global, recv_counts, recv_displs] =
      comm.gatherv(kmp::send_buf(succ), kmp::recv_counts_out(), kmp::recv_displs_out());
  std::vector<idx_t> dist_global;
  if (comm.is_root()) {
    dist_global.resize(succ_global.size());
  }
  comm.scatterv(kmp::send_buf(dist_global), kmp::recv_buf(dist),
                kmp::send_counts(recv_counts), kmp::send_displs(recv_displs),
                kmp::recv_count(static_cast<int>(succ.size())));
}
}  // namespace kascade
