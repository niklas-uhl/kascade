module Config
value_paths = Dict(
    "p" => ["config", "num_ranks"],
    "kagen_option_string" => ["config", "input", "kagen_option_string"],
    "commit" => ["config", "git_tag"],
    "time" => ["timer"],
    "algorithm" => ["config", "algorithm"],
    "async_caching" => ["config", "async_pointer_chasing", "use_caching"],
    "pointer_doubling_aggregation_level" => ["config", "pointer_doubling", "aggregation_level"],
    "pointer_doubling_use_local_preprocessing" => ["config", "pointer_doubling", "use_local_preprocessing"]
)

timer_value_paths = Dict(
    "total_time" => ["root", "ranking", "max"],
)

first_iteration = 1

end
