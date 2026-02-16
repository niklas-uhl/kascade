#pragma once

#include <kamping/mpi_datatype.hpp>

namespace karam::mpi {
template <typename>
struct datatype {
    static MPI_Datatype get_type() = delete;
};

template <typename T>
concept BuiltinMPIDatatype = kamping::mpi_type_traits<T>::is_builtin;

template <typename T>
concept MPIDatatype = BuiltinMPIDatatype<T> || requires(T t) {
    { datatype<T>::get_type() } -> std::same_as<MPI_Datatype>;
};

template <typename Range, typename T>
concept MPIBuffer =
    MPIDatatype<T> && std::ranges::contiguous_range<Range> && std::same_as<T, std::ranges::range_value_t<Range>>;
} // namespace karam::mpi

template <karam::mpi::BuiltinMPIDatatype T>
struct karam::mpi::datatype<T> {
    static MPI_Datatype get_type() {
        return kamping::mpi_datatype<T>();
    }
};
