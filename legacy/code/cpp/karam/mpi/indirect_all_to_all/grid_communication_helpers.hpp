#pragma once

#include <cmath>

#include <kamping/communicator.hpp>

namespace karam::mpi {

/// @brief Augments a plain message with its (initial) source and destination PE
/// TODO discuss if this is too much overhead in our use cases
template <typename PayloadType>
class IndirectMessage {
public:
  using Payload = PayloadType;
  IndirectMessage() {}

  IndirectMessage(PayloadType payload) : _source{0u}, _destination{0u}, _payload{payload} {}

  IndirectMessage(std::uint32_t source, std::uint32_t destination, PayloadType payload)
    : _source{source},
      _destination{destination},
      _payload{payload} {}

  void set_source(std::uint32_t source) {
    _source = source;
  }

  void set_destination(std::uint32_t destination) {
    _destination = destination;
  }

  [[nodiscard]] std::uint32_t get_source() const {
    return _source;
  }

  [[nodiscard]] std::uint32_t get_destination() const {
    return _destination;
  }
  void swap_source_and_destination() {
    std::swap(_source, _destination);
  }

  Payload const& payload() const {
    return _payload;
  }

  friend std::ostream& operator<<(std::ostream& out, IndirectMessage<PayloadType> const& msg) {
    return out << "(source: " << msg.get_source() << ", destination: " << msg.get_destination()
               << ", payload: " << msg.payload() << ")";
  }

private:
  std::uint32_t _source;
  std::uint32_t _destination;
  Payload       _payload;
};

/// @brief Represent a two dimensional grid communicator
///
/// PEs are row-major and abs(#row, #columns) <= 1
/// 0  1  2  3
/// 4  5  6  7
/// 8  9  10 11
/// 12 13 14 15
///
/// If #PE != #row * #column then the PEs of the last incomplete row are transposed and appended to
/// the first rows and do not form an own row based communicator 0  1  2  3 (16) 4  5  6  7 (17) 8
/// 9  10 11 12 13 14 15 16 17
class GridCommunicator {
public:
  GridCommunicator(kamping::Communicator<> const& comm = kamping::comm_world()) {
    double const      sqrt       = std::sqrt(comm.size());
    const std::size_t floor_sqrt = static_cast<std::size_t>(std::floor(sqrt));
    const std::size_t ceil_sqrt  = static_cast<std::size_t>(std::ceil(sqrt));
    // if size exceeds the threshold we can afford one more column
    const std::size_t threshold = floor_sqrt * ceil_sqrt;
    _number_columns = (static_cast<std::size_t>(comm.size()) < threshold) ? floor_sqrt : ceil_sqrt;
    const std::size_t num_pe_in_small_column = comm.size() / _number_columns;
    std::size_t       row_num                = proxy_row_index(comm.rank());
    std::size_t       column_num             = proxy_col_index(comm.rank());
    _size_complete_rectangle                 = _number_columns * num_pe_in_small_column;
	
    if (comm.rank() >= _size_complete_rectangle) {
      row_num = comm.rank() % _number_columns; // virtual group
    }
    // TODO replace with communicator creation based on MPI_COMM_CREATE
    _row_comm    = comm.split(static_cast<int>(row_num), comm.rank_signed());
    _column_comm = comm.split(static_cast<int>(column_num), comm.rank_signed());
	
	//von hier mein code für allgather
	if (std::lround(std::pow(2, std::lround(std::log2(comm.size())))) == comm.size())
	{
		my_comm_are_disjoint = true;
		
		std::uint32_t my_number_columns = std::lround(std::pow(2, std::lround(std::log2(comm.size())) / 2));
		
		std::uint32_t my_row_num = comm.rank() / my_number_columns;
		std::uint32_t my_col_num = comm.rank() % my_number_columns;

		my_row_comm = comm.split(my_row_num, comm.rank_signed());
		my_col_comm = comm.split(my_col_num, comm.rank_signed());
	}
	
	
	
	//bis hier
  }
  
  template<typename packet>
  std::vector<packet> allgatherv(kamping::Communicator<>& comm, std::vector<packet>& send_buf)
  {
	  if (my_comm_are_disjoint)
	  {
		  std::vector<packet> zwischen_ergebnis;
		  my_row_comm.allgatherv(kamping::send_buf(send_buf), kamping::recv_buf<kamping::resize_to_fit>(zwischen_ergebnis));
		  std::vector<packet> finales_ergebnis;
		  my_col_comm.allgatherv(kamping::send_buf(zwischen_ergebnis), kamping::recv_buf<kamping::resize_to_fit>(finales_ergebnis));
		  return finales_ergebnis;
	  }
	  else
	  {
		  std::vector<packet> finales_ergebnis;
		  comm.allgatherv(kamping::send_buf(send_buf), kamping::recv_buf<kamping::resize_to_fit>(finales_ergebnis));
		  return finales_ergebnis;
	  }
  }
  
  [[nodiscard]] std::size_t proxy_row_index(std::size_t destination_rank) const {
    return destination_rank / _number_columns;
  }
  [[nodiscard]] std::size_t proxy_col_index(std::size_t destination_rank) const {
    return destination_rank % _number_columns;
  }
  [[nodiscard]] std::size_t rank_within_column_comm(std::size_t global_rank) const {
    return proxy_row_index(global_rank);
  }
  [[nodiscard]] std::size_t rank_within_row_comm(std::size_t global_rank) const {
    bool const is_in_complete_rectangle = global_rank < _size_complete_rectangle;
    return is_in_complete_rectangle ? proxy_col_index(global_rank) : _number_columns;
  }
  [[nodiscard]] kamping::Communicator<> const& row_comm() const {
    return _row_comm;
  }
  [[nodiscard]] kamping::Communicator<> const& col_comm() const {
    return _column_comm;
  }

private:
  std::size_t             _size_complete_rectangle;
  std::size_t             _number_columns;
  kamping::Communicator<> _row_comm;
  kamping::Communicator<> _column_comm;
  

  bool my_comm_are_disjoint = false;
  kamping::Communicator<> my_row_comm, my_col_comm;
  
};

class GridExchangeHelper {
public:
  GridExchangeHelper(GridCommunicator const& grid_comm) : _grid_comm{grid_comm} {}
  [[nodiscard]] std::size_t first_pass_destination(std::size_t rank_final_destination) const {
    return _grid_comm.proxy_col_index(rank_final_destination);
  }
  [[nodiscard]] std::size_t second_pass_destination(std::size_t rank_final_destination) const {
    return _grid_comm.proxy_row_index(rank_final_destination);
  }
  void first_level_exchange() {}

private:
  GridCommunicator const& _grid_comm;
};
} // namespace karam::mpi
