#include <kamping/collectives/barrier.hpp>
#include <kamping/environment.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/spdlog_adapter/logging.hpp>
#include <spdlog/cfg/env.h>

#include "detail/algorithm_factory.hpp"
#include "detail/generation.hpp"
#include "detail/parsing.hpp"
#include "spdlog/stopwatch.h"

auto main(int argc, char* argv[]) -> int {
  {
    kamping::Environment const env;

    spdlog::cfg::load_env_levels();
    kamping::logging::setup_logging();

    kamping::Communicator const comm;

    // IO
    auto config = parse_args(std::span{argv, static_cast<std::size_t>(argc)});

    auto G = generate_input(config, comm);
    comm.barrier();

    //// reference implementation
    // spdlog::stopwatch stopwatch;
    // auto reference_impl = BFSCC(comm);
    // if (config.verify_level > 0) {
    //   reference_impl.ingest(G);
    //   reference_impl.run();
    //   SPDLOG_LOGGER_INFO(spdlog::get("root"),
    //                      "Reference: Found {} components in {} secs.",
    //                      reference_impl.get_num_components(), stopwatch);
    // }

    // actual benchmark runs
    // Report report;
    for (std::size_t i = 0; i < config.iterations; i++) {
      comm.barrier();
      spdlog::stopwatch stopwatch;
      auto algo = get_algorithm(config, comm);
      kamping::measurements::timer().synchronize_and_start("ingest_graph");
      algo->ingest(G);
      kamping::measurements::timer().stop_and_append();

      kamping::measurements::timer().synchronize_and_start("ranking");
      algo->run();
      kamping::measurements::timer().stop_and_append();

      kamping::measurements::timer().synchronize_and_start("extract_component_count");
      kamping::measurements::timer().stop_and_append();
      SPDLOG_LOGGER_INFO(spdlog::get("root"), "Finished run in {} secs.", stopwatch);

      // verify(G, reference_impl, *algo, config.verify_level,
      //        config.verify_continue_on_mismatch, comm);

      // report.push_stats("components", [&] {
      //   auto component_sizes = kacc::component_sizes(G, algo->get_labeling(), comm);
      //   SPDLOG_LOGGER_TRACE(spdlog::get("gather"), "component_sizes: {}",
      //                       component_sizes);
      //   return kacc::ComponentStats(component_sizes, comm);
      // });
      // report.step_iteration();
    }

    //// Print our results
    // if (kamping::comm_world().is_root()) {
    //   auto out = make_output_stream(config.output_path);
    //   report.push_config(config);
    //   report.print(*out);
    // }
  }
  return 0;
}
