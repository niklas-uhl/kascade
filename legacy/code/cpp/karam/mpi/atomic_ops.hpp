#pragma once
#include <mpi.h>
#include <type_traits>

namespace karam::mpi::ops {
    template<typename Op>
    concept AtomicOp = requires(Op op) {
    {Op::mpi_op()} -> std::same_as<MPI_Op>;
};
/// @brief builtin maximum operation (aka `MPI_MAX`)
struct max_impl {
    static MPI_Op mpi_op() {
        return MPI_MAX;
    }
};
constexpr max_impl max{};

/// @brief builtin minimum operation (aka `MPI_MIN`)
struct min_impl {
    static MPI_Op mpi_op() {
        return MPI_MIN;
    }
};
constexpr min_impl min{};

/// @brief builtin summation operation (aka `MPI_SUM`)
struct add_impl {
    static MPI_Op mpi_op() {
        return MPI_SUM;
    }
};
constexpr add_impl add{};

/// @brief builtin multiply operation (aka `MPI_PROD`)
struct multiply_impl {
    static MPI_Op mpi_op() {
        return MPI_PROD;
    }
};
constexpr multiply_impl multiply{};

/// @brief builtin logical and operation (aka `MPI_LAND`)
struct logical_and_impl {
    static MPI_Op mpi_op() {
        return MPI_LAND;
    }
};
constexpr logical_and_impl logical_and{};

/// @brief builtin bitwise and operation (aka `MPI_BAND`)
struct bit_and_impl {
    static MPI_Op mpi_op() {
        return MPI_BAND;
    }
};
constexpr bit_and_impl bit_and{};

/// @brief builtin logical or operation (aka `MPI_LOR`)
struct logical_or_impl {
    static MPI_Op mpi_op() {
        return MPI_LOR;
    }
};
constexpr logical_or_impl logical_or{};

/// @brief builtin bitwise or operation (aka `MPI_BOR`)
struct bit_or_impl {
    static MPI_Op mpi_op() {
        return MPI_BOR;
    }
};
constexpr bit_or_impl bit_or{};

/// @brief builtin logical xor operation (aka `MPI_LXOR`)
struct logical_xor_impl {
    static MPI_Op mpi_op() {
        return MPI_LXOR;
    }
};
constexpr logical_xor_impl logical_xor{};

/// @brief builtin bitwise xor operation (aka `MPI_BXOR`)
struct bit_xor_impl {
    static MPI_Op mpi_op() {
        return MPI_BXOR;
    }
};
constexpr bit_xor_impl bit_xor{};

/// @brief builtin no op operation (aka `MPI_NO_OP`)
struct no_op_impl {
    static MPI_Op mpi_op() {
        return MPI_NO_OP;
    }
};
constexpr no_op_impl no_op{};

/// @brief builtin replace operation (aka `MPI_REPLACE`)
struct replace_impl {
    static MPI_Op mpi_op() {
        return MPI_REPLACE;
    }
};
constexpr replace_impl replace{};
} // namespace karam::mpi::ops
