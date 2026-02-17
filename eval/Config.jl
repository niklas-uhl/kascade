module Config
value_paths = Dict(
    "p" => ["config", "num_ranks"],
    "kagen_option_string" => ["config", "input", "kagen_option_string"],
    "commit" => ["config", "git_tag"],
    "time" => ["timer"],
    "algorithm" => ["config", "algorithm"],
    "async_caching" => ["config", "async_pointer_chasing", "use_caching"],
    "pointer_doubling_aggregation_level" => ["config", "pointer_doubling", "aggregation_level"],
    "pointer_doubling_use_local_preprocessing" => ["config", "pointer_doubling", "use_local_preprocessing"],
    "sparse_ruling_set_sync" => ["config", "sparse_ruling_set", "sync"],
    "sparse_ruling_set_sync_locality_aware" => ["config", "sparse_ruling_set", "sync_locality_aware"],
    "sparse_ruling_set_spawn" => ["config", "sparse_ruling_set", "spawn"],
    "sparse_ruling_set_dehne_factor" => ["config", "sparse_ruling_set", "dehne_factor"],
    "sparse_ruling_set_sanders_factor" => ["config", "sparse_ruling_set", "sanders_factor"],
    "sparse_ruling_set_ruler_selection" => ["config", "sparse_ruling_set", "ruler_selection"],
    "sparse_ruling_set_briefkasten_local_threshold" => ["config", "sparse_ruling_set", "briefkasten", "local_threshold"],
    "sparse_ruling_set_briefkasten_poll_skip_threshold" => ["config", "sparse_ruling_set", "briefkasten", "poll_skip_threshold"],
    "eulertour_algorithm" => ["config", "euler_tour", "algorithm"],
    "eulertour_use_high_degree_handling" => ["config", "euler_tour", "use_high_degree_handling"],
    "rma_pointer_chasing_sync_mode" => ["config", "rma_pointer_chasing", "sync_mode"],
    "rma_pointer_chasing_batch_size" => ["config", "rma_pointer_chasing", "batch_size"],
)

timer_value_paths = Dict(
    "total_time" => ["root", "ranking", "max"],
)

first_iteration = 1

end
