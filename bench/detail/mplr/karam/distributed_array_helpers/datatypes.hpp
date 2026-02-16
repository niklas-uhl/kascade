#pragma once

#include <cstddef>
#include <ostream>

#include "karam/mpi/datatype.hpp"

namespace karam {
template <typename Data>
struct IndexData {
  std::size_t index;
  Data        data;
  IndexData() = default;
  IndexData(std::size_t index_, Data const& data_) : index{index_}, data{data_} {}
  friend std::ostream& operator<<(std::ostream& out, IndexData<Data> const& index_data) {
    return out << "Index: " << index_data.index << " Data: " << index_data.data;
  }
};

template <typename Data>
struct AddressData {
  std::uintptr_t address;
  Data           data;
  AddressData() = default;
  AddressData(std::size_t address_, Data const& data_) : address{address_}, data{data_} {}
  friend std::ostream& operator<<(std::ostream& out, AddressData<Data> const& address_data) {
    return out << "Address: " << address_data.address << " Data: " << address_data.data;
  }
};

struct IndexAddress {
  std::size_t    index;
  std::uintptr_t address;
  IndexAddress() = default;
  IndexAddress(std::size_t index_, std::uintptr_t const& address_)
    : index{index_},
      address{address_} {}

  template <typename T>
  AddressData<T> create_reply(T const& t) const {
    return AddressData<T>{index, t};
  }

  friend std::ostream& operator<<(std::ostream& out, IndexAddress const& elem) {
    return out << "(idx: " << elem.index << " address: " << elem.address << ")";
  }
};

template <typename T>
struct karam::mpi::datatype<IndexData<T>> {
  static MPI_Datatype get_type() {
    return kamping::mpi_datatype<IndexData<T>>();
  }
};

template <>
struct karam::mpi::datatype<IndexAddress> {
  static MPI_Datatype get_type() {
    return kamping::mpi_datatype<IndexAddress>();
  }
};

template <typename T>
struct karam::mpi::datatype<AddressData<T>> {
  static MPI_Datatype get_type() {
    return kamping::mpi_datatype<AddressData<T>>();
  }
};

template <typename Data>
inline bool operator<(IndexData<Data> const& lhs, IndexData<Data> const& rhs) {
  return lhs.index < rhs.index;
}

template <typename Data>
inline bool operator==(IndexData<Data> const& lhs, IndexData<Data> const& rhs) {
  return lhs.index == rhs.index && lhs.data == rhs.data;
}
} // namespace karam
