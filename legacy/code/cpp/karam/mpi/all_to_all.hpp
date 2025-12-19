#pragma once

#include <ranges>
#include <span>

#include <absl/container/flat_hash_map.h>
#include <kamping/collectives/barrier.hpp>
#include <kamping/communicator.hpp>
#include <kamping/status.hpp>
#include <mpi.h>

#include "karam/mpi/communication_concepts.hpp"
#include "karam/mpi/datatype.hpp"
#include "karam/mpi/indirect_all_to_all/grid_all_to_all_impl.hpp"
#include "karam/utils/containers.hpp"
#include "karam/utils/debug_print.hpp"
#include "karam/utils/non_copyable.hpp"

namespace karam::mpi {

template <typename Context, typename T>
concept MessageContext = requires(Context ctx, std::vector<T>& buf) {
                           { ctx.receive_into(buf) };
                           { ctx.recv_count_signed() } -> std::same_as<int>;
                           { ctx.recv_count() } -> std::same_as<std::size_t>;
                         };

template <typename T>
class SelfMessage {
public:
  SelfMessage(std::span<T const> msg) : _msg(msg) {}
  void receive_into(MPIBuffer<T> auto& recv_buffer) const {
    KASSERT(std::ssize(recv_buffer) >= recv_count_signed());
    std::ranges::copy_n(_msg.begin(), recv_count_signed(), std::data(recv_buffer));
  }

  int recv_count_signed() const {
    return static_cast<int>(_msg.size());
  }

  std::size_t recv_count() const {
    return _msg.size();
  }

private:
  std::span<T const> _msg;
  static_assert(MessageContext<SelfMessage<T>, T>);
};

template <MPIDatatype T>
class ProbedMessage : utils::NonCopyable {
public:
  void receive_into(MPIBuffer<T> auto& recv_buffer) const {
    KASSERT(recv_buffer.size() >= recv_count_signed());
    MPI_Datatype datatype = karam::mpi::datatype<T>::get_type();
    MPI_Recv(
      std::data(recv_buffer),
      recv_count_signed(),
      datatype,
      _status.MPI_SOURCE,
      _status.MPI_TAG,
      _comm.mpi_communicator(),
      MPI_STATUS_IGNORE
    );
  }

  int recv_count_signed() const {
    MPI_Datatype datatype = karam::mpi::datatype<T>::get_type();
    int          count    = 0;
    MPI_Get_count(&_status, datatype, &count);
    return count;
  }

  std::size_t recv_count() const {
    return static_cast<std::size_t>(recv_count_signed());
  }

  ProbedMessage(MPI_Status& status, kamping::Communicator<> const& comm)
    : _status(status),
      _comm(comm) {}

private:
  MPI_Status&                    _status;
  kamping::Communicator<> const& _comm;
};

template <typename Functor, typename T>
concept ReceiveHandler = requires(Functor f, SelfMessage<T> const& message) {
                           { f(message) };
                         };

template <typename InputRange, typename T = send_buffer_list_datatype<InputRange>>
  requires SendBufferList<InputRange, T>
void sparse_all_to_all(
  InputRange const&              send_buf,
  ReceiveHandler<T> auto&&       on_message,
  kamping::Communicator<> const& comm = kamping::comm_world(),
  int                            tag  = 0
) {
  std::size_t messages_to_send = 0;
  for (auto const& kv: send_buf) {
    if (std::size(kv.second) > 0) {
      if (kv.first == comm.rank_signed()) {
        SelfMessage<T> self_message{std::span(kv.second)};
        on_message(self_message);
      } else {
        messages_to_send++;
      }
    }
  }
  std::vector<MPI_Request> requests(messages_to_send);
  auto                     request = requests.begin();
  for (auto const& kv: send_buf) {
    MPI_Datatype datatype = karam::mpi::datatype<T>::get_type();
    if (std::size(kv.second) == 0 || kv.first == comm.rank_signed()) {
      continue;
    }
    MPI_Issend(
      std::data(kv.second),
      static_cast<int>(std::size(kv.second)),
      datatype,
      kv.first,
      tag,
      comm.mpi_communicator(),
      &*request
    );
    request++;
  }
  MPI_Status  status;
  MPI_Request barrier_request = MPI_REQUEST_NULL;
  while (true) {
    int got_message = false;
    MPI_Iprobe(MPI_ANY_SOURCE, tag, comm.mpi_communicator(), &got_message, &status);
    if (got_message) {
      ProbedMessage<T> probed_message{status, comm};
      on_message(probed_message);
    }
    if (barrier_request != MPI_REQUEST_NULL) {
      int barrier_finished = false;
      MPI_Test(&barrier_request, &barrier_finished, MPI_STATUS_IGNORE);
      if (barrier_finished) {
        break;
      }
    } else {
      int all_sends_finished = false;
      MPI_Testall(
        static_cast<int>(requests.size()),
        requests.data(),
        &all_sends_finished,
        MPI_STATUSES_IGNORE
      );
      if (all_sends_finished) {
        MPI_Ibarrier(comm.mpi_communicator(), &barrier_request);
      }
    }
  }
  comm.barrier();
}

template <typename InputRange, typename T = send_buffer_list_datatype<InputRange>>
  requires SendBufferList<InputRange, T>
auto grid_mpi_all_to_all(InputRange const& send_buf, GridCommunicator const& grid_comm) {
  auto mpi_result_rowwise  = rowwise_exchange(send_buf, grid_comm);
  auto rowwise_recv_buf    = mpi_result_rowwise.extract_recv_buffer();
  auto rowwise_recv_counts = mpi_result_rowwise.extract_recv_counts();
  auto rowwise_recv_displs = mpi_result_rowwise.extract_recv_displs();

  return columnwise_exchange(rowwise_recv_buf, grid_comm);
}

template <typename SendBuffer, typename DestinationHandler>
auto grid_mpi_all_to_all(
  SendBuffer const&       send_buf,
  DestinationHandler&&    get_final_destination,
  GridCommunicator const& grid_comm
) {
  auto mpi_result_rowwise  = rowwise_exchange<true>(send_buf, get_final_destination, grid_comm);
  auto rowwise_recv_buf    = mpi_result_rowwise.extract_recv_buffer();
  auto rowwise_recv_counts = mpi_result_rowwise.extract_recv_counts();
  auto rowwise_recv_displs = mpi_result_rowwise.extract_recv_displs();
	std::cout << "richtig funktion" << std::endl;
  return columnwise_exchange(rowwise_recv_buf, grid_comm);
}


} // namespace karam::mpi
