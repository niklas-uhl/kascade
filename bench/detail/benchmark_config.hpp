#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <kamping/communicator.hpp>

#include "git.h"
#include "input/generation.hpp"
#include "kascade/configuration.hpp"

enum class Algorithm : std::uint8_t {
  GatherChase,
  PointerDoubling,
  AsyncPointerDoubling,
  RMAPointerDoubling,
  invalid,
};

struct Config {
  std::size_t num_ranks = kamping::world_size();
  // std::string hostname = KACC_HOSTNAME;
  // std::string os = KACC_OS;
  // std::string processor = KACC_PROCESSOR;
  std::string git_tag = std::string(git::CommitSHA1().substr(0, 8)) +
                        (git::AnyUncommittedChanges() ? "-dirty" : "");
  std::string output_path = "stdout";
  std::size_t iterations = 1;
  kascade::input::Config input;
  Algorithm algorithm = Algorithm::invalid;
  kascade::AsyncPointerChasingConfig async_pointer_chasing;
  std::size_t verify_level = 1;
  bool verify_continue_on_mismatch = false;
};
