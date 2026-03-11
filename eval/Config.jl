module Config
value_paths = Dict(
    "p" => ["config", "num_ranks"],
    "kagen_option_string" => ["config", "input", "kagen_option_string"],
    "i_mpi_adjust_alltoallv" => ["config", "i_mpi_adjust_alltoallv"],
    "commit" => ["config", "git_tag"],
    "time" => ["timer"],
    "algorithm" => ["config", "algorithm"],
    "async_caching" => ["config", "async_pointer_chasing", "use_caching"],
    "pointer_doubling_aggregation_level" => ["config", "pointer_doubling", "aggregation_level"],
    "pointer_doubling_use_local_preprocessing" => ["config", "pointer_doubling", "use_local_preprocessing"],
    "pointer_doubling_grid_comm" => ["config", "pointer_doubling", "use_grid_communication"],
    "pointer_doubling_grid_communicator_mode" => ["config", "pointer_doubling", "grid_communicator_mode"],
    "sparse_ruling_set_grid_communicator_mode" => ["config", "sparse_ruling_set", "grid_communicator_mode"],
    "sparse_ruling_set_base_algorithm" => ["config", "sparse_ruling_set", "base_algorithm"],
    "sparse_ruling_set_sparse_ruling_set_rounds" => ["config", "sparse_ruling_set", "sparse_ruling_set_rounds"],
    "sparse_ruling_set_sync" => ["config", "sparse_ruling_set", "sync"],
    "sparse_ruling_set_sync_locality_aware" => ["config", "sparse_ruling_set", "sync_locality_aware"],
    "sparse_ruling_set_spawn" => ["config", "sparse_ruling_set", "spawn"],
    "sparse_ruling_set_no_precompute_rulers" => ["config", "sparse_ruling_set", "no_precompute_rulers"],
    "sparse_ruling_set_post_invert" => ["config", "sparse_ruling_set", "post_invert"],
    "sparse_ruling_set_post_invert_detect_leaves" => ["config", "sparse_ruling_set", "post_invert_detect_leaves"],
    "sparse_ruling_set_cache_owners" => ["config", "sparse_ruling_set", "cache_owners"],
    "sparse_ruling_set_root_gather_threshold" => ["config", "sparse_ruling_set", "root_gather_threshold"],
    "sparse_ruling_set_grid_comm" => ["config", "sparse_ruling_set", "use_grid_communication"],
    "sparse_ruling_set_dehne_factor" => ["config", "sparse_ruling_set", "dehne_factor"],
    "sparse_ruling_set_sanders_factor" => ["config", "sparse_ruling_set", "sanders_factor"],
    "sparse_ruling_set_ultimate_factor" => ["config", "sparse_ruling_set", "ultimate_factor"],
    "sparse_ruling_set_ultimate_sanders_factor" => ["config", "sparse_ruling_set", "ultimate_sanders_factor"],
    "sparse_ruling_set_round_limit" => ["config", "sparse_ruling_set", "round_limit"],
    "sparse_ruling_set_ruler_selection" => ["config", "sparse_ruling_set", "ruler_selection"],
    "sparse_ruling_set_ruler_propagation_mode" => ["config", "sparse_ruling_set", "ruler_propagation_mode"],
    "sparse_ruling_set_ruler_propagation_use_aggregation" => ["config", "sparse_ruling_set", "use_aggregation_in_ruler_propagation"],
    "sparse_ruling_set_use_local_contraction" => ["config", "sparse_ruling_set", "use_local_contraction"],
    "sparse_ruling_set_reverse_list_locality_aware" => ["config", "sparse_ruling_set", "reverse_list_locality_aware"],
    "sparse_ruling_set_briefkasten_local_threshold" => ["config", "sparse_ruling_set", "briefkasten", "local_threshold"],
    "sparse_ruling_set_briefkasten_poll_skip_threshold" => ["config", "sparse_ruling_set", "briefkasten", "poll_skip_threshold"],
    "eulertour_algorithm" => ["config", "euler_tour", "algorithm"],
    "eulertour_use_high_degree_handling" => ["config", "euler_tour", "use_high_degree_handling"],
    "rma_pointer_chasing_sync_mode" => ["config", "rma_pointer_chasing", "sync_mode"],
    "rma_pointer_chasing_batch_size" => ["config", "rma_pointer_chasing", "batch_size"],
)

timer_value_paths = Dict(
    "total_time" => ["root", "ranking", "max"],
    "base_case" => ["root", "ranking", "base_case", "max"], 
    "chase_rulers" => ["root", "ranking", "chase_ruler", "max"],
    "base_case_chase_rulers" => ["root", "ranking", "base_case", "chase_ruler", "max"],
    "base_case_base_case" => ["root", "ranking", "base_case", "base_case", "max"],
    "base_case_ruler_propagation" => ["root", "ranking", "base_case", "ruler_propagation", "max"],
    "base_case_post_invert" => ["root", "ranking", "base_case", "post_invert", "max"],
    "base_case_pack_base_case" => ["root", "ranking", "base_case", "pack_base_case", "max"],
    "base_case_unpack_base_case" => ["root", "ranking", "base_case", "unpack_base_case", "max"],
    "init_node_type" => ["root", "ranking", "init_node_type", "max"],
    "precompute_ruler_permutation" => ["root", "ranking", "precompute_ruler_permutation", "max"],
    "find_rulers" => ["root", "ranking", "find_rulers", "max"],
    "invert_list" => ["root", "ranking", "invert_list", "max"],
    "cache_owners_ruler_prop" => ["root", "ranking", "ruler_propagation", "cache_owners", "max"],
    "find_leaves" => ["root", "ranking", "find_leaves", "max"],
    "ruler_propagation" => ["root", "ranking", "ruler_propagation", "max"],
    "pack_base_case" => ["root", "ranking", "pack_base_case", "max"],
    "unpack_base_case" => ["root", "ranking", "unpack_base_case", "max"],
    "post_invert" => ["root", "ranking", "post_invert", "max"],
    "fixup_unreached" => ["root", "ranking", "fixup_unreached", "max"],
    "local_contraction" => ["root", "ranking", "local_contraction", "max"],
    "local_uncontraction" => ["root", "ranking", "local_uncontraction", "max"],
)

first_iteration = 1


function format_pointer_doubling(;kwargs...)
    parts = ["PointerDoubling"]
    if kwargs[:pointer_doubling_use_local_preprocessing]
        push!(parts, "+preprocessing")
    end
    if kwargs[:pointer_doubling_aggregation_level] != nothing && kwargs[:pointer_doubling_aggregation_level] != "none"
        push!(parts, "agg=$(kwargs[:pointer_doubling_aggregation_level])")
    end
    use_grid_comm = get(kwargs, :pointer_doubling_grid_comm, false)
    if use_grid_comm
        grid_mode = get(kwargs, :pointer_doubling_grid_communicator_mode, "topology-aware")
        push!(parts, " [comm=$(grid_mode)]")
    end
    return join(parts, " ")
end

function format_rma_pointer_doubling(;kwargs...)
    parts = ["RMAPointerDoubling"]
    push!(parts, "(sync_mode=$(kwargs[:rma_pointer_chasing_sync_mode])")
    if kwargs[:rma_pointer_chasing_sync_mode] == "passive_target"
        push!(parts, " batch_size=$(kwargs[:rma_pointer_chasing_batch_size])")
    end
    push!(parts, ")")
    return join(parts)
end

function format_briefkasten_params(local_threshold, poll_skip_threshold)
    isnothing(local_threshold) && return ""
    isnothing(poll_skip_threshold) && return ""
    return "(Δ=$local, ρ=$poll)"
end

function format_ruler_selection_params(;kwargs...)
    selection = kwargs[:sparse_ruling_set_ruler_selection]
    param = ""
    if selection == "dehne"
        param = "γ=$(kwargs[:sparse_ruling_set_dehne_factor])"
    elseif selection == "sanders"
        param = "γ=$(kwargs[:sparse_ruling_set_sanders_factor])"
    elseif selection == "ultimate"
        param = "γ=$(kwargs[:sparse_ruling_set_ultimate_factor])"
    elseif selection == "ultimate-sanders"
        param = "γ=$(kwargs[:sparse_ruling_set_ultimate_sanders_factor])"
    elseif selection == "limit-rounds"
        param = "rounds=$(kwargs[:sparse_ruling_set_round_limit])"
    end
    return "ruler_selection=$(selection)($(param))"
end

function format_sparse_ruling_set(;kwargs...)
    parts = ["SparseRulingSet"]
    base_algorithm =  kwargs[:sparse_ruling_set_base_algorithm]
    rounds = get(kwargs, :sparse_ruling_set_sparse_ruling_set_rounds, base_algorithm == "SparseRulingSet" ? 2 : 1)
    if (rounds >= 3 && base_algorithm == "SparseRulingSet") || (rounds >= 2 && base_algorithm != "SparseRulingSet")
        push!(parts, "-$rounds-level")
    end
    push!(parts, "-$(base_algorithm)")
    sync = kwargs[:sparse_ruling_set_sync]
    if sync
        push!(parts, "-sync")
    else
        push!(parts, "-async")
    end
    locality_aware = get(kwargs, :sparse_ruling_set_sync_locality_aware, false)
    if locality_aware && sync
        push!(parts, "-locality-aware")
    end
    local_contraction = get(kwargs, :sparse_ruling_set_use_local_contraction, false)
    if local_contraction
        push!(parts, "-local-contraction")
    end
    spawn = get(kwargs, :sparse_ruling_set_spawn, false)
    if spawn
        push!(parts, "-spawn")
    end
    use_grid_comm = get(kwargs, :sparse_ruling_set_grid_comm, false)
    if use_grid_comm
        grid_mode = get(kwargs, :sparse_ruling_set_grid_communicator_mode, "topology-aware")
        push!(parts, " [comm=$(grid_mode)]")
    end
    
    push!(parts, " [$(format_ruler_selection_params(;kwargs...))]")
    alltoallv_variant = get(kwargs, :i_mpi_adjust_alltoallv, "");
    if alltoallv_variant != ""
        push!(parts, " I_MPI_ADJUST_ALLTOALLV=$(alltoallv_variant)")
    end
    return join(parts)
end

const CONFIG_FORMATTERS = Dict(
    "PointerDoubling" => format_pointer_doubling,
    "SparseRulingSet" => format_sparse_ruling_set,
    "RMAPointerDoubling" => format_rma_pointer_doubling,
)

function to_config_name(;kwargs...)
    # algo = kwargs[:algorithm]
    algorithm = kwargs[:algorithm]
    if haskey(CONFIG_FORMATTERS, algorithm)
        return CONFIG_FORMATTERS[algorithm](;kwargs...)
    end
    return algorithm
    # name = "$(algorithm)"
    # if algorithm == "EulerTour"
    #     name *=  "-$(kwargs[:eulertour_algorithm])"
    #     algorithm = kwargs[:eulertour_algorithm] # fallback to the actual algorithm for further naming
    # end
    # if algorithm == "AsyncPointerDoubling" && kwargs[:async_caching]
    #     name *= "+cache"
    # end
    # if algorithm == "PointerDoubling"
    #     if kwargs[:pointer_doubling_use_local_preprocessing]
    #         name *= "+preprocessing"
    #     end
    #     name *= " agg=$(kwargs[:pointer_doubling_aggregation_level])"
    # end
    # if algorithm == "SparseRulingSet"
    #     name *= "-$(kwargs[:sparse_ruling_set_base_algorithm])"
    #     rounds = kwargs[:sparse_ruling_set_sparse_ruling_set_rounds]
    #     if kwargs[:sparse_ruling_set_base_algorithm] == "SparseRulingSet" && rounds .!= nothing && rounds .>= 3
    #         name *= "-$rounds-level"
    #     end
    #     # elseif kwargs[:sparse_ruling_set_base_algorithm] == nothing
    #     #     name *= "-old"
    #     # end
    #     async = !kwargs[:sparse_ruling_set_sync]
    #     if !async
    #         name *= "-sync"
    #         if kwargs[:sparse_ruling_set_sync_locality_aware] == true
    #             name *= "-locality-aware"
    #         end
    #     else
    #         name *= "-async"
    #     end
    #     if kwargs[:sparse_ruling_set_grid_comm] == true
    #         name *= "-grid-comm"
    #     end
    #     if kwargs[:sparse_ruling_set_spawn] == true
    #         name *= "-spawn"
    #     end
    #     name *= " (ruler_selection=$(kwargs[:sparse_ruling_set_ruler_selection])"
    #     if kwargs[:sparse_ruling_set_ruler_selection] == "dehne"
    #         name *= ", dehne_factor=$(kwargs[:sparse_ruling_set_dehne_factor])"
    #     elseif kwargs[:sparse_ruling_set_ruler_selection] == "sanders"
    #         name *= ", sanders_factor=$(kwargs[:sparse_ruling_set_sanders_factor])"
    #     elseif kwargs[:sparse_ruling_set_ruler_selection] == "limit-rounds"
    #         name *= ", round_limit=$(kwargs[:sparse_ruling_set_round_limit])"
    #     elseif kwargs[:sparse_ruling_set_ruler_selection] == "ultimate"
    #         name *= ", round_limit=$(kwargs[:sparse_ruling_set_ultimate_factor])"
    #     end
    #     name *= ")"
    #     if kwargs[:sparse_ruling_set_ruler_propagation_mode] != "pull"
    #         name *= " (ruler_propagation_mode=$(kwargs[:sparse_ruling_set_ruler_propagation_mode]))"
    #     end
    #     if kwargs[:sparse_ruling_set_ruler_propagation_use_aggregation] == true
    #         name *= " (ruler_prop_agg)"
    #     end
    #     if async
    #         name *= " $(format_briefkasten_params(kwargs[:sparse_ruling_set_briefkasten_local_threshold], kwargs[:sparse_ruling_set_briefkasten_poll_skip_threshold]))"
    #     end
    # end
    # if algorithm == "RMAPointerDoubling"
    #     name *= " (sync_mode=$(kwargs[:rma_pointer_chasing_sync_mode])"
    #     if kwargs[:rma_pointer_chasing_sync_mode] == "passive_target"
    #         name *= ", batch_size=$(kwargs[:rma_pointer_chasing_batch_size])"
    #     end
    #     name *= ")"
    # end
    return name
end

end
