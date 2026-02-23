#pragma once

#include <cstdint>

#include "kascade/types.hpp"

namespace kascade {

enum class NodeType : std::uint8_t { root, leaf, ruler, unreached, reached };

struct RulerMessage {
  idx_t target_idx;
  idx_t ruler;
  rank_t dist_from_ruler;
};
}  // namespace kascade
