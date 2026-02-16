#pragma once

#include <numeric>
#include <ostream>
#include <vector>

#include <kamping/collectives/allreduce.hpp>
#include <kamping/communicator.hpp>
#include <karam/mpi/all_to_all.hpp>
#include <karam/mpi/rdma.hpp>
#include <mpi.h>

#include "ips4o.hpp"
#include "karam/distributed_array_helpers/grid_based_read_requests.hpp"
#include "karam/utils/local_chunk_distribution_strategies.hpp"
#include "karam/utils/timer.hpp"

namespace karam {

///@brief Object encapsulates some stats on number of send and write requests and provides
/// functionality to collect
/// these values from all PEs.
/// TODO accumulation function should potentially moved to Timer/More General class
struct Stats {
  std::size_t number_read_requests         = 0u;
  std::size_t number_write_requests        = 0u;
  std::size_t number_posted_read_requests  = 0u;
  std::size_t number_posted_write_requests = 0u;
  std::size_t number_recv_read_requests    = 0u;
  std::size_t number_recv_write_requests   = 0u;
  Stats       accumulate_local_stats(kamping::Communicator<> const& comm) const {
    Stats accu_stats      = *this;
    auto  allreduce_value = [&](auto& elem) {
      elem = comm.allreduce(kamping::send_buf({elem}), kamping::op(kamping::ops::plus<>()))
               .extract_recv_buffer()
               .front();
    };
    allreduce_value(accu_stats.number_read_requests);
    allreduce_value(accu_stats.number_write_requests);
    allreduce_value(accu_stats.number_posted_read_requests);
    allreduce_value(accu_stats.number_posted_write_requests);
    allreduce_value(accu_stats.number_recv_read_requests);
    allreduce_value(accu_stats.number_recv_write_requests);
    return accu_stats;
  }
  friend std::ostream& operator<<(std::ostream& out, Stats const& stats) {
    return out << "{read_req=" << stats.number_read_requests
               << " write_req=" << stats.number_write_requests
               << " posted_read=" << stats.number_posted_read_requests
               << " posted_write_req=" << stats.number_posted_write_requests
               << " recv_read_req=" << stats.number_recv_read_requests
               << " recv_write_req=" << stats.number_recv_write_requests << "}";
  }
};

template <typename T>
class DistributedArray {
public:
  using value_type = T;
  DistributedArray(std::size_t total_size, kamping::Communicator<> const& comm)
    : _comm{comm},
      _chunk_distribution_strategy{comm.rank_signed(), comm.size_signed(), total_size},
      _win{mpi::Win<T>(_chunk_distribution_strategy.get_local_chunk_size(), false, comm)},
      _lock_all{_win.lock_all()},
      _total_size{total_size} {}

  void atomic_write(std::size_t /*index*/, value_type const& /*value*/) {
    // TODO unimplemented
  }
  void write(std::size_t index, value_type const& value) {
    const std::size_t rank = _chunk_distribution_strategy.get_rank_owning_index(index);
    const std::size_t target_displacement = _chunk_distribution_strategy.get_local_index_of(index);
    _lock_all.on(static_cast<int>(rank)).put(target_displacement, value);
    ++_stats.number_write_requests;
    ++_stats.number_posted_write_requests;
  }
  void atomic_read(std::size_t /*index*/, value_type& /*value*/) {
    // TODO unimplemented
  }
  void read(std::size_t index, value_type& value) {
    const std::size_t rank = _chunk_distribution_strategy.get_rank_owning_index(index);
    const std::size_t target_displacement = _chunk_distribution_strategy.get_local_index_of(index);

    _lock_all.on(static_cast<int>(rank)).get(target_displacement, value);
    ++_stats.number_read_requests;
    ++_stats.number_posted_read_requests;
  }
  void lock_step() {
    _lock_all.unlock();
    _lock_all.lock();
  }
  void globally_synchronized_lock_step() {
    lock_step();
    _comm.barrier();
  }
  std::size_t total_size() {
    return _total_size;
  }
  std::vector<T> gather_on_rank(std::size_t rank) {
    std::vector<T> data;
    if (_comm.rank() == rank) {
      data.resize(total_size());
      for (std::size_t i = 0; i < total_size(); ++i) {
        read(i, data[i]);
      }
    }
    globally_synchronized_lock_step();
    return data;
  }

  Stats& stats() {
    return _stats;
  }
  std::span<T> local_chunk_distributed_array() {
    return _win.local_data();
  }

private:
  kamping::Communicator<> const& _comm;
  LocalChunkSizeHelper           _chunk_distribution_strategy;
  mpi::Win<T>                    _win;
  mpi::LockAll<T>                _lock_all;
  std::size_t                    _total_size;
  Stats                          _stats;
};

template <typename T>
class FencedDistributedArray {
public:
  using value_type = T;
  FencedDistributedArray(std::size_t total_size, kamping::Communicator<> const& comm)
    : _comm{comm},
      _win{mpi::Win<T>(total_size, false, comm)},
      _epoch{_win.fence()},
      _total_size{total_size},
      _chunk_size{(total_size / comm.size()) + 1} {}

  void atomic_write(std::size_t /*index*/, value_type const& /*value*/) {
    // TODO unimplemented
  }
  void write(std::size_t index, value_type const& value) {
    const std::size_t rank                = get_rank(index);
    const std::size_t target_displacement = get_target_displacement(rank, index);
    _epoch.on(static_cast<int>(rank)).put(target_displacement, value);
  }
  void atomic_read(std::size_t /*index*/, value_type& /*value*/) {
    // TODO unimplemented
  }
  void read(std::size_t index, value_type& value) {
    const std::size_t rank                = get_rank(index);
    const std::size_t target_displacement = get_target_displacement(rank, index);
    _epoch.on(static_cast<int>(rank)).get(target_displacement, value);
  }

  void globally_synchronized_lock_step() {
    _epoch = _epoch.fence();
  }
  std::size_t total_size() {
    return _total_size;
  }
  std::vector<T> gather_on_rank(std::size_t rank) {
    std::vector<T> data;
    if (_comm.rank() == rank) {
      data.resize(total_size());
      for (std::size_t i = 0; i < total_size(); ++i) {
        read(i, data[i]);
      }
    }
    globally_synchronized_lock_step();
    return data;
  }

private:
  std::size_t get_rank(std::size_t index) {
    return index / _chunk_size;
  }
  std::size_t get_target_displacement(std::size_t rank, std::size_t index) {
    return index - (rank * _chunk_size);
  }

  kamping::Communicator<> const& _comm;
  mpi::Win<T>                    _win;
  mpi::AccessEpoch<T>            _epoch;
  std::size_t                    _total_size;
  std::size_t                    _chunk_size;
};

enum class RequestCombinationLevel { no_combination, local_combination, general_combination };

template <typename T, typename ExchangeMethod, typename DistributionManager>
class DistributedArrayWithMessages {
public:
  using value_type = T;

  DistributedArrayWithMessages(
    std::size_t                    total_size,
    RequestCombinationLevel             deduplication_level,
    kamping::Communicator<> const& comm
  )
    : _comm{comm},
      _distribution_manager{comm.rank_signed(), comm.size_signed(), total_size},
      _total_size{total_size},
      _local_part_distributed_array(_distribution_manager.get_local_chunk_size()),
      _deduplication_level{deduplication_level} {
    // utils::debug_print(_local_part_distributed_array.size(), " local size:");
  }

  void write(std::size_t index, value_type const& value) {
    _write_requests.emplace_back(index, value);
  }

  void read(std::size_t index, value_type& value) {
    _read_requests.emplace_back(index, reinterpret_cast<std::uintptr_t>(std::addressof(value)));
  }
  auto& local_chunk_distributed_array() {
    return _local_part_distributed_array;
  }

  void globally_synchronized_lock_step() {
    _stats.number_read_requests += _read_requests.size();
    _stats.number_write_requests += _write_requests.size();
    static_cast<ExchangeMethod&>(*this).handle_write_requests();
    static_cast<ExchangeMethod&>(*this).handle_read_requests();
  }
  std::size_t total_size() {
    return _total_size;
  }
  std::vector<T> gather_on_rank(std::size_t rank) {
    std::vector<T> data;
    if (_comm.rank() == rank) {
      data.resize(total_size());
      for (std::size_t i = 0; i < total_size(); ++i) {
        read(i, data[i]);
      }
    }
    globally_synchronized_lock_step();
    return data;
  }
  Stats& stats() {
    return _stats;
  }

protected:
  kamping::Communicator<> const&                     _comm;
  mpi::GridCommunicator                              _grid_comm;
  DistributionManager                                _distribution_manager;
  std::unordered_map<int, std::vector<IndexData<T>>> _write_request_for_target_pe;
  utils::default_init_vector<IndexAddress>           _read_requests;
  utils::default_init_vector<IndexData<T>>           _write_requests;
  std::unordered_map<int, std::span<IndexAddress>>   _read_request_for_target_pe;
  std::size_t                                        _total_size;
  utils::default_init_vector<T>                      _local_part_distributed_array;
  RequestCombinationLevel                                 _deduplication_level;
  Stats                                              _stats;

  bool perform_local_request_combination() const {
    switch (_deduplication_level) {
      case RequestCombinationLevel::no_combination:
        return false;
      case RequestCombinationLevel::local_combination:
        return true;
      case RequestCombinationLevel::general_combination:
        return true;
    }
    return false;
  }

  bool perform_global_request_combination() const {
    switch (_deduplication_level) {
      case RequestCombinationLevel::no_combination:
        return false;
      case RequestCombinationLevel::local_combination:
        return false;
      case RequestCombinationLevel::general_combination:
        return true;
    }
    return false;
  }

  template <typename Requests>
  auto sort_requests_per_receiver(Requests& requests) {
    auto sort_by_receiver_and_index = [&](auto const& lhs, auto const& rhs) {
      const auto receiver_lhs = _distribution_manager.get_signed_rank_owning_index(lhs.index);
      const auto receiver_rhs = _distribution_manager.get_signed_rank_owning_index(rhs.index);
      return std::tie(receiver_lhs, lhs.index) < std::tie(receiver_rhs, rhs.index);
    };
    ips4o::sort(requests.begin(), requests.end(), sort_by_receiver_and_index);
  }
  template <typename Requests>
  auto bucket_sort_requests_per_receiver(Requests& requests) {
    std::vector<std::size_t> counts(_comm.size());
    for (auto const& request: requests) {
      ++counts[_distribution_manager.get_signed_rank_owning_index(request.index)];
    }
    auto counts2 = counts;
    std::exclusive_scan(counts.begin(), counts.end(), counts.begin(), 0ull);
    Requests requests2(requests.size());
    for (auto const& request: requests) {
      requests2[counts[_distribution_manager.get_signed_rank_owning_index(request.index)]++] =
        request;
    }
    auto sort_by_index = [&](auto const& lhs, auto const& rhs) {
      return lhs.index < rhs.index;
    };

    for (std::size_t i = 0; i < counts.size(); ++i) {
      std::sort(requests2.begin() + counts2[i], requests2.begin() + counts[i], sort_by_index);
    }
    requests = std::move(requests2);
  }
  template <typename Requests>
  auto deduplicate_requests(Requests& sorted_requests) {
    if (sorted_requests.empty()) {
      return sorted_requests;
    }
    auto is_index_equal = [](auto const& lhs, auto const& rhs) {
      return lhs.index == rhs.index;
    };
    std::size_t number_unique_requests = 1ull;
    for (std::size_t i = 1; i < sorted_requests.size(); ++i) {
      number_unique_requests += !is_index_equal(sorted_requests[i - 1], sorted_requests[i]);
    }
    // if (static_cast<double>(number_unique_requests) / sorted_request.size() >
    // deduplication_threshold) {
    //     return sorted_requests;
    // }
    Requests unique_requests(number_unique_requests);
    std::unique_copy(
      sorted_requests.begin(),
      sorted_requests.end(),
      unique_requests.begin(),
      is_index_equal
    );
    return unique_requests;
  }

  template <typename Requests>
  void deduplicate_requests_inplace(Requests& sorted_requests) {
    auto is_index_equal = [](auto const& lhs, auto const& rhs) {
      return lhs.index == rhs.index;
    };
    std::size_t number_unique_requests = 0ull;
    for (std::size_t i = 1; i < sorted_requests.size(); ++i) {
      number_unique_requests += !is_index_equal(sorted_requests[i - 1], sorted_requests[i]);
    }
    // if (static_cast<double>(number_unique_requests) / sorted_request.size() >
    // deduplication_threshold) {
    //     return;
    // }
    auto it = std::unique(sorted_requests.begin(), sorted_requests.end(), is_index_equal);
    sorted_requests.erase(it, sorted_requests.end());
  }

  template <typename Container>
  auto create_messages_from_sorted_data(Container const& sorted_requests_in_contiguous_memory) {
    using Data = typename Container::value_type;
    std::unordered_map<int, std::span<Data const>> messages;
    if (sorted_requests_in_contiguous_memory.empty()) {
      return messages;
    }
    auto         begin                = sorted_requests_in_contiguous_memory.begin();
    std::int64_t begin_cur_message    = 0;
    std::int64_t end_cur_message      = 0;
    int          receiver_cur_message = _distribution_manager.get_signed_rank_owning_index(
      sorted_requests_in_contiguous_memory.front().index
    );
    for (std::size_t i = 1; i < sorted_requests_in_contiguous_memory.size(); ++i) {
      auto const& elem     = sorted_requests_in_contiguous_memory[i];
      auto const  index    = elem.index;
      auto const  receiver = _distribution_manager.get_signed_rank_owning_index(index);
      // utils::debug_print(std::vector<int>{i, receiver}, "i receiver");
      if (receiver != receiver_cur_message) {
        end_cur_message = static_cast<std::int64_t>(i);
        messages.emplace(
          receiver_cur_message,
          std::span(begin + begin_cur_message, begin + end_cur_message)
        );
        begin_cur_message = end_cur_message;
      }
      receiver_cur_message = receiver;
    }
    messages.emplace(
      receiver_cur_message,
      std::span(begin + begin_cur_message, sorted_requests_in_contiguous_memory.end())
    );
    return messages;
  }

  template <typename Container>
  auto create_send_counts_from_sorted_data(Container const& sorted_requests_in_contiguous_memory) {
    std::vector<int> send_counts(this->_comm.size(), 0);
    auto receiver_message = create_messages_from_sorted_data(sorted_requests_in_contiguous_memory);
    for (auto const& [receiver, message]: receiver_message) {
      send_counts[static_cast<std::size_t>(receiver)] = static_cast<int>(message.size());
    }
    return send_counts;
  }

  ///@brief Iterates over the initial read requests and fills in the requested data for duplicated
  /// requests that have
  /// previously been filtered out.
  /// The method relies on the fact that the order of the request is fix and the first request from
  /// a range of requests with same index is served.
  template <typename Container>
  void copy_data_from_first_request_in_range(Container const& container) {
    if (container.empty()) {
      return;
    }
    auto is_index_equal = [](auto const& lhs, auto const& rhs) {
      return lhs.index == rhs.index;
    };

    T data = *reinterpret_cast<T*>(container.front().address);
    for (std::size_t i = 1; i < container.size(); ++i) {
      auto& prev_elem = container[i - 1];
      auto& elem      = container[i];
      if (is_index_equal(prev_elem, elem)) {
        T* ptr = reinterpret_cast<T*>(elem.address);
        *ptr   = data;
      } else {
        data = *reinterpret_cast<T*>(container[i].address);
      }
    }
  }
};

template <typename T>
class DataBufferSenderContainer {
public:
  DataBufferSenderContainer(std::span<T> recv_buffer, std::span<int> displacements)
    : _recv_buffer{recv_buffer},
      _displacements{displacements} {}

  class Iterator {
  public:
    using BufferIterator = typename std::span<T>::iterator;

    Iterator(std::span<T> recv_buffer, std::span<int> displacements, std::size_t index)
      : _recv_buffer{recv_buffer},
        _displacements{displacements},
        _index_recv_buf{0},
        _index_displs{1} {
      advance_to(index);
    }
    void advance_to(std::size_t index_into_recv_buffer) {
      _index_recv_buf = index_into_recv_buffer;
      for (; _index_displs < _displacements.size()
             && static_cast<std::size_t>(_displacements[_index_displs]) <= _index_recv_buf;
           ++_index_displs)
        ;
    }

    Iterator& operator++() {
      advance_to(++_index_recv_buf);
      return *this;
    }
    std::pair<BufferIterator, int> operator*() {
      return std::pair<BufferIterator, int>(
        _recv_buffer.begin() + static_cast<int>(_index_recv_buf),
        static_cast<int>(_index_displs - 1)
      );
    }

    friend bool operator!=(Iterator const& lhs, Iterator const& rhs) {
      return lhs._recv_buffer.begin() != rhs._recv_buffer.begin()
             || lhs._index_recv_buf != rhs._index_recv_buf;
    }

    std::span<T>   _recv_buffer;
    std::span<int> _displacements;
    std::size_t    _index_recv_buf;
    std::size_t    _index_displs;
  };

  auto begin() {
    return Iterator(_recv_buffer, _displacements, 0);
  }

  auto end() {
    return Iterator(_recv_buffer, _displacements, _recv_buffer.size());
  }

private:
  std::span<T>   _recv_buffer;
  std::span<int> _displacements;
};

template <typename T>
class DistributedArrayWithGridAllToAll : public DistributedArrayWithMessages<
                                           T,
                                           DistributedArrayWithGridAllToAll<T>,
                                           LocalChunkSizeHelper> {
public:
  using DistributedArrayWithMessages<T, DistributedArrayWithGridAllToAll<T>, LocalChunkSizeHelper>::
    DistributedArrayWithMessages;
  void handle_write_requests() {
    // write request via grid all to all
    this->sort_requests_per_receiver(this->_write_requests);
    if (this->perform_local_request_combination()) {
      this->deduplicate_requests_inplace(this->_write_requests);
    }
    this->_stats.number_posted_write_requests += this->_write_requests.size();
    auto receiver_message = this->create_messages_from_sorted_data(this->_write_requests);
    auto result           = mpi::grid_mpi_all_to_all(receiver_message, this->_grid_comm);
    dump(std::move(this->_write_requests));
    auto recv_buffer = result.extract_recv_buffer();
    this->_stats.number_recv_write_requests += recv_buffer.size();
    for (auto const& recv_elem: recv_buffer) {
      auto const& elem  = recv_elem.payload();
      auto const& index = elem.index;
      auto const& data  = elem.data;
      this->_local_part_distributed_array[this->_distribution_manager.get_local_index_of(index)] =
        data;
    }
  }

  void handle_read_requests() {
    // preprocess send requests (possible deduplication etc.), send them and prepare replies
    // this is done in a block so that allocated memory is freed as soon it is no longer required.

    karam::get_timer().start("read_local_preprocessing");
    auto const& preprocessed_requests =
      this->perform_local_request_combination() ? deduplicate_read_requests() : this->_read_requests;
    this->_stats.number_posted_read_requests += preprocessed_requests.size();
    karam::get_timer().stop("read_local_preprocessing");

    auto get_index = [](IndexAddress const& elem) {
      return elem.index;
    };
    auto get_destination = [&](std::size_t const& index) {
      return this->_distribution_manager.get_signed_rank_owning_index(index);
    };

    if (this->perform_global_request_combination()) {
      handle_read_requests_with_request_combination(
        preprocessed_requests,
        get_index,
        get_destination
      );
    } else {
      handle_read_requests_without_request_combination(
        preprocessed_requests,
        get_index,
        get_destination
      );
    }
    if (this->perform_local_request_combination()) {
      this->copy_data_from_first_request_in_range(this->_read_requests);
    }
    // karam::get_timer().stop("read_write_request_in_memory");
    this->_read_requests.clear();
    dump(std::move(this->_read_requests));
  }
  auto get_num_requested_elements() const {
    return num_requested_elements;
  }

private:
  std::size_t num_requested_elements;
  auto        deduplicate_read_requests() {
    this->sort_requests_per_receiver(this->_read_requests);
    return this->deduplicate_requests(this->_read_requests);
  }
  template <typename Requests, typename GetIndex, typename GetDestination>
  void handle_read_requests_with_request_combination(
    Requests&& preprocessed_requests, GetIndex&& get_index, GetDestination&& get_destination
  ) {
    auto create_reply = [&](std::size_t const& index) {
      auto const& data =
        this->_local_part_distributed_array[this->_distribution_manager.get_local_index_of(index)];
      ++this->_stats.number_recv_read_requests;
      return IndexData<T>{index, data};
    };
    auto recv_buffer_replies = handle_read_requests_in_grid_with_filter<T, std::size_t>(
                                 preprocessed_requests,
                                 this->_grid_comm,
                                 get_index,
                                 get_destination,
                                 create_reply
    )
                                 .extract_recv_buffer();

    for (auto const& recv_elem: recv_buffer_replies) {
      T* ptr = reinterpret_cast<T*>(recv_elem.address);
      *ptr   = recv_elem.data;
    }
  }
  template <typename Requests, typename GetIndex, typename GetDestination>
  void handle_read_requests_without_request_combination(
    Requests&& preprocessed_requests, GetIndex&& get_index, GetDestination&& get_destination
  ) {
    auto create_reply = [&](IndexAddress const& index_address) {
      auto const& data =
        this->_local_part_distributed_array[this->_distribution_manager
                                              .get_local_index_of(index_address.index)];
      ++this->_stats.number_recv_read_requests;
      return AddressData<T>{index_address.address, data};
    };
    auto recv_buffer_replies = handle_read_requests_in_grid<T, std::size_t>(
                                 preprocessed_requests,
                                 this->_grid_comm,
                                 get_index,
                                 get_destination,
                                 create_reply
    )
                                 .extract_recv_buffer();
    for (auto const& recv_elem: recv_buffer_replies) {
      T* ptr = reinterpret_cast<T*>(recv_elem.payload().address);
      *ptr   = recv_elem.payload().data;
    }
  }
};

template <typename T>
class DistributedArrayWithPlainAllToAll : public DistributedArrayWithMessages<
                                            T,
                                            DistributedArrayWithPlainAllToAll<T>,
                                            LocalChunkSizeHelper> {
public:
  using DistributedArrayWithMessages<
    T,
    DistributedArrayWithPlainAllToAll<T>,
    LocalChunkSizeHelper>::DistributedArrayWithMessages;
  void handle_write_requests() {
    // write request via grid all to all
    this->sort_requests_per_receiver(this->_write_requests);
    if (this->perform_local_request_combination()) {
      this->deduplicate_requests_inplace(this->_write_requests);
    }
    this->_stats.number_posted_write_requests += this->_write_requests.size();
    auto send_counts = this->create_send_counts_from_sorted_data(this->_write_requests);
    auto result      = this->_comm.alltoallv(
      kamping::send_buf(this->_write_requests),
      kamping::send_counts(send_counts)
    );
    dump(std::move(this->_write_requests));
    auto recv_buffer = result.extract_recv_buffer();
    this->_stats.number_recv_write_requests += recv_buffer.size();
    for (auto const& recv_elem: recv_buffer) {
      auto const& index = recv_elem.index;
      auto const& data  = recv_elem.data;
      this->_local_part_distributed_array[this->_distribution_manager.get_local_index_of(index)] =
        data;
    }
  }
  void handle_read_requests() {
    this->sort_requests_per_receiver(this->_read_requests);
    auto const& preprocessed_requests = this->perform_local_request_combination()
                                          ? this->deduplicate_requests(this->_read_requests)
                                          : this->_read_requests;

    this->_stats.number_posted_read_requests += preprocessed_requests.size();
    auto send_counts = this->create_send_counts_from_sorted_data(preprocessed_requests);
    auto result      = this->_comm.alltoallv(
      kamping::send_buf(preprocessed_requests),
      kamping::send_counts(send_counts)
    );
    auto recv_buffer        = result.extract_recv_buffer();
    auto recv_counts        = result.extract_recv_counts();
    auto recv_displacements = result.extract_recv_displs();
    auto inclusive_recv_displacements =
      recv_displacements; // used to determine the index at which an item should be written in the
                          // reply buffer
                          //
    this->_stats.number_recv_read_requests += recv_buffer.size();
    utils::default_init_vector<AddressData<T>> reply_buffer(recv_buffer.size());
    auto                                       recv_buffer_sender =
      DataBufferSenderContainer(std::span(recv_buffer), std::span(recv_displacements));
    for (auto [it_recv_buf, sender]: recv_buffer_sender) {
      auto const& index   = it_recv_buf->index;
      auto const& address = it_recv_buf->address;
      auto const& data =
        this->_local_part_distributed_array[this->_distribution_manager.get_local_index_of(index)];
      auto const& receiver = static_cast<std::size_t>(sender);
      reply_buffer[static_cast<std::size_t>(inclusive_recv_displacements[receiver]++)] =
        AddressData(address, data);
    }
    auto replies = this->_comm.alltoallv(
      kamping::send_buf(reply_buffer),
      kamping::send_counts(recv_counts),
      kamping::send_displs(recv_displacements)
    );
    auto recv_reply_buffer = replies.extract_recv_buffer();
    for (auto const& recv_reply: recv_reply_buffer) {
      T* ptr = reinterpret_cast<T*>(recv_reply.address);
      *ptr   = recv_reply.data;
    }
    if (this->perform_local_request_combination()) {
      this->copy_data_from_first_request_in_range(this->_read_requests);
    }
    dump(std::move(this->_read_requests));
  }

private:
  auto make_write_requests_contiguous() {
    std::vector<int> send_counts(this->_comm.size(), 0);
    for (auto const& [rank, data]: this->_write_request_for_target_pe) {
      send_counts[static_cast<std::size_t>(rank)] = static_cast<int>(data.size());
    }
    auto send_displacements = send_counts;
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displacements.begin(), 0);
    utils::default_init_vector<IndexData<T>> send_buffer(
      static_cast<std::size_t>(send_displacements.back() + send_counts.back())
    );
    for (auto const& [dest_rank, data]: this->_write_request_for_target_pe) {
      std::copy(
        data.begin(),
        data.end(),
        send_buffer.begin() + send_displacements[static_cast<std::size_t>(dest_rank)]
      );
    }
    return std::make_tuple(
      std::move(send_buffer),
      std::move(send_counts),
      std::move(send_displacements)
    );
  }
  auto make_read_requests_contiguous() {
    std::vector<int> send_counts(this->_comm.size(), 0);
    for (auto const& [rank, data]: this->_read_request_for_target_pe) {
      send_counts[static_cast<std::size_t>(rank)] = static_cast<int>(data.size());
    }
    auto send_displacements = send_counts;
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displacements.begin(), 0);
    const std::size_t num_send_elems =
      static_cast<std::size_t>(send_displacements.back() + send_counts.back());
    utils::default_init_vector<IndexAddress> send_buffer(num_send_elems);
    for (auto const& [dest_rank, data]: this->_read_request_for_target_pe) {
      std::copy(
        data.begin(),
        data.end(),
        send_buffer.begin() + send_displacements[static_cast<std::size_t>(dest_rank)]
      );
    }
    return std::make_tuple(
      std::move(send_buffer),
      std::move(send_counts),
      std::move(send_displacements)
    );
  }
};
} // namespace karam
