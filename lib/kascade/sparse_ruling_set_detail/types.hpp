#pragma once

#include <cstdint>
#include <string_view>

#include "kascade/types.hpp"

namespace kascade {

enum class NodeType : std::uint8_t { root, leaf, ruler, unreached, reached };

inline auto format_as(NodeType type) -> std::string_view {
  switch (type) {
    case NodeType::root:
      return "root";
    case NodeType::leaf:
      return "leaf";
    case NodeType::ruler:
      return "ruler";
    case NodeType::unreached:
      return "unreached";
    case NodeType::reached:
      return "reached";
      break;
  }
  std::unreachable();
}

struct RulerMessage {
  idx_t target_idx;
  idx_t ruler;
  rank_t dist_from_ruler;
};
}  // namespace kascade
