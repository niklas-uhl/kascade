#include "generation.hpp"

#include <kagen.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

auto generate_input(Config const& /*config*/, kamping::Communicator<> const& comm)
    -> std::vector<std::size_t> {
  kagen::KaGen gen(comm.mpi_communicator());
  gen.EnableBasicStatistics();
  gen.UseCSRRepresentation();
  spdlog::stopwatch stopwatch;
  //auto G = gen.GenerateFromOptionString(config.kagen_option_string);
  // TODO think about right input format
  spdlog::get("root")->info("Generation finished in {} seconds.", stopwatch);
  return {};
}
