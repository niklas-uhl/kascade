#pragma once

#include <kassert/kassert.hpp>

namespace kascade::assert {
  constexpr int normal = kassert::assert::normal;
  constexpr int with_communication = 100;
}
