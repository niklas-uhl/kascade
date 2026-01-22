#include <kamping/collectives/barrier.hpp>
#include <kamping/environment.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/spdlog_adapter/logging.hpp>
#include <spdlog/cfg/env.h>

#include "detail/algorithm_factory.hpp"
#include "detail/algorithm_impl.hpp"
#include "detail/parsing.hpp"
#include "detail/reporting.hpp"
#include "detail/serialization.hpp"  // IWYU pragma: keep
#include "detail/verification.hpp"
#include "spdlog/stopwatch.h"

auto main(int argc, char* argv[]) -> int {
  {
    kamping::Environment const env;

    spdlog::cfg::load_env_levels();
    kamping::logging::setup_logging();

    kamping::Communicator const comm;

    // IO
    auto config = parse_args(std::span{argv, static_cast<std::size_t>(argc)});

    std::vector<kascade::idx_t> succ = kascade::input::generate_input(config.input, comm);
    comm.barrier();

    // reference implementation
    spdlog::stopwatch stopwatch;
    auto reference_impl = GatherRank(comm);
    if (config.verify_level > 0) {
      reference_impl.ingest(succ);
      reference_impl.run();
      SPDLOG_LOGGER_INFO(spdlog::get("root"), "Reference: Finished run in {} secs.",
                         stopwatch);
    }

    // actual benchmark runs
    Report report;
    for (std::size_t i = 0; i < config.iterations; i++) {
      comm.barrier();
      spdlog::stopwatch stopwatch;
      auto algo = get_algorithm(config, comm);
      kamping::measurements::timer().synchronize_and_start("ingest_graph");
      algo->ingest(succ);
      kamping::measurements::timer().stop_and_append();

      kamping::measurements::timer().synchronize_and_start("ranking");
      algo->run();
      kamping::measurements::timer().stop_and_append();
      SPDLOG_LOGGER_INFO(spdlog::get("root"), "Finished run in {} secs.", stopwatch);

      verify(succ, reference_impl, *algo, config.verify_level,
             config.verify_continue_on_mismatch, comm);
      report.step_iteration();
    }

    // Print our results
    if (kamping::comm_world().is_root()) {
      auto out = make_output_stream(config.output_path);
      report.push_config(config);
      report.print(*out);
    }
  }
  return 0;
}
