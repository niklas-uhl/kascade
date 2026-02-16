#pragma once

#include <cstdint>

namespace karam {

///@brief Class defining how the global array is distributed over the PEs. In this basic setting each PE gets n / p
/// elements and the last PE additional n % p elements.
class LocalChunkSizeHelper {
public:
    LocalChunkSizeHelper(int rank, int size, std::size_t total_size) : _rank{rank}, _size{size} {
        _regular_local_chunk_size = total_size / static_cast<std::size_t>(size);
        auto const remainder      = total_size % static_cast<std::size_t>(size);
        bool const is_last_pe     = _rank == (_size - 1);
        _local_chunk_size         = is_last_pe ? (_regular_local_chunk_size + remainder) : _regular_local_chunk_size;
    }
    std::size_t get_rank_owning_index(std::size_t global_index) const {
        const std::size_t last_rank = static_cast<std::size_t>(_size - 1);
        if (global_index < _regular_local_chunk_size * last_rank) {
            return global_index / _regular_local_chunk_size;
        }
        return last_rank;
    }
    int get_signed_rank_owning_index(std::size_t global_index) const {
        return static_cast<int>(get_rank_owning_index(global_index));
    }
    std::size_t get_local_index_of(std::size_t global_index) const {
        return global_index - (get_rank_owning_index(global_index) * _regular_local_chunk_size);
    }
    std::size_t get_local_chunk_size() const {
        return _local_chunk_size;
    }

private:
    int         _rank;
    int         _size;
    std::size_t _local_chunk_size;
    std::size_t _regular_local_chunk_size;
};
} // namespace karam
