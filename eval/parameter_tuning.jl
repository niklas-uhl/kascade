using DataFrames
using DataFramesMeta
using Statistics
using AlgebraOfGraphics
using CairoMakie

include("KascadeEval.jl")
import .KascadeEval

# dirs = [
#     # "./data/i10/sparse-ruling-set_26_02_12/",
#     # "./data/i10/sparse-ruling-set-spawn-factor_26_02_12/",
#     # "./data/horeka/pointer-doubling_26_02_06/",
#     # "./data/horeka/gather-chase_26_02_06/",
#     # "./data/supermuc/sync-sparse-ruling-set_26_02_09/",
#     # "./data/supermuc/sparse-ruling-set_26_02_09/",
#     # "./data/supermuc/sparse-ruling-set-spawn-factor_26_02_10/",
#     # "./data/supermuc/tree-ruling-set_26_02_12/",
#     # "./data/supermuc/eulertour-sparse-ruling-set_26_02_12/",
#     # "./data/i10/tree-ruling-set_26_02_12/",
#     # "./data/supermuc/sync-sparse-ruling-set_26_02_09/",
#     # "./data/supermuc/sync-sparse-ruling-set-locality-aware_26_02_12/",
#     # "./data/supermuc/async-sparse-ruling-set_26_02_12/",
#     # "./data/supermuc/sparse-ruling-set_26_02_09/",
#     # "./data/supermuc/sparse-ruling-set-spawn-factor_26_02_10/",
#     # "./data/supermuc/rma-pointer-doubling_26_02_16/"
#     # "./data/supermuc/sparse-ruling-set-spawn-sanders_26_02_16/",
#     # "./data/supermuc/sparse-ruling-set-spawn-sanders-others_26_02_16/",
#     # "./data/supermuc/sparse-ruling-set-spawn-sanders-briefkasten-tuning_26_02_16/",
#     # "./data/supermuc/sparse-ruling-set-spawn-sanders-owner-cache_26_02_16/",
#     # "./data/supermuc/sparse-ruling-set-two-level_26_02_19/",
#     # "./data/supermuc/sparse-ruling-set-grid-a2a_26_02_19/",
#     # "./data/horeka/sparse-ruling-set-two-level_26_02_18/",
#     # "./data/horeka/sparse-ruling-set-spawn-sanders-bla_26_02_18/"
#     # "./data/supermuc/sparse-ruling-set-spawn_26_02_09/",
# ]

# -------------------------
# Data loading & transform
# -------------------------
algo_params = [
    :algorithm,
    :async_caching,
    :pointer_doubling_aggregation_level,
    :pointer_doubling_use_local_preprocessing,
    :sparse_ruling_set_base_algorithm,
    :sparse_ruling_set_sync,
    :sparse_ruling_set_sync_locality_aware,
    :sparse_ruling_set_grid_comm,
    :sparse_ruling_set_spawn,
    :sparse_ruling_set_ruler_selection,
    :sparse_ruling_set_ruler_propagation_mode,
    :sparse_ruling_set_dehne_factor,
    :sparse_ruling_set_sanders_factor,
    :sparse_ruling_set_briefkasten_local_threshold,
    :sparse_ruling_set_briefkasten_poll_skip_threshold,
    :rma_pointer_chasing_sync_mode,
    :rma_pointer_chasing_batch_size,
    :eulertour_algorithm,
    :commit
]
function format_briefkasten_params(local_threshold, poll_skip_threshold)
    if local_threshold == nothing || poll_skip_threshold == nothing
        return ""
    else
        return "(Δ=$(local_threshold), ρ=$(poll_skip_threshold))"
    end
end
function to_config_name(;kwargs...)
    algorithm = kwargs[:algorithm]
    name = "$(algorithm)"
    if algorithm == "EulerTour"
        name *=  "-$(kwargs[:eulertour_algorithm])"
        algorithm = kwargs[:eulertour_algorithm] # fallback to the actual algorithm for further naming
    end
    if algorithm == "AsyncPointerDoubling" && kwargs[:async_caching]
        name *= "+cache"
    end
    if algorithm == "PointerDoubling"
        if kwargs[:pointer_doubling_use_local_preprocessing]
            name *= "+preprocessing"
        end
        name *= " agg=$(kwargs[:pointer_doubling_aggregation_level])"
    end
    if algorithm == "SparseRulingSet"
        name *= "-$(kwargs[:sparse_ruling_set_base_algorithm])"
        # if kwargs[:sparse_ruling_set_base_algorithm] == "SparseRulingSet"
        #     name *= "-two-level"
        # elseif kwargs[:sparse_ruling_set_base_algorithm] == nothing
        #     name *= "-old"
        # end
        async = !kwargs[:sparse_ruling_set_sync]
        if !async
            name *= "-sync"
            if kwargs[:sparse_ruling_set_sync_locality_aware] == true
                name *= "-locality-aware"
            end
            if kwargs[:sparse_ruling_set_grid_comm] == true
                name *= "-grid-comm"
            end
        else
            name *= "-async"
        end
        if kwargs[:sparse_ruling_set_spawn] == true
            name *= "-spawn"
        end
        name *= " (ruler_selection=$(kwargs[:sparse_ruling_set_ruler_selection])"
        if kwargs[:sparse_ruling_set_ruler_selection] == "dehne"
            name *= ", dehne_factor=$(kwargs[:sparse_ruling_set_dehne_factor])"
        # elseif kwargs[:sparse_ruling_set_ruler_selection] == "sanders"
        #     name *= ", sander_factor=$(kwargs[:sparse_ruling_set_sanders_factor])"
        end
        name *= ")"
        if kwargs[:sparse_ruling_set_ruler_propagation_mode] != "pull"
            name *= " (ruler_propagation_mode=$(kwargs[:sparse_ruling_set_ruler_propagation_mode]))"
        end
        if async
            name *= " $(format_briefkasten_params(kwargs[:sparse_ruling_set_briefkasten_local_threshold], kwargs[:sparse_ruling_set_briefkasten_poll_skip_threshold]))"
        end
    end
    if algorithm == "RMAPointerDoubling"
        name *= " (sync_mode=$(kwargs[:rma_pointer_chasing_sync_mode])"
        if kwargs[:rma_pointer_chasing_sync_mode] == "passive_target"
            name *= ", batch_size=$(kwargs[:rma_pointer_chasing_batch_size])"
        end
        name *= ")"
    end
    return name
end

# -------------------------
# Plotting
# -------------------------


axis = (; xscale=log2)
facet = (; linkyaxes=:none)
figure = (; size=(2000, 1200))

dirs = [
    # "./data/supermuc/sparse-ruling-set-grid-a2a_26_02_19/",
    # "./data/supermuc/sparse-ruling-set-two-level_26_02_19/"
    # "./data/supermuc/async-tuning_26_02_21/"
    # "./data/supermuc/async-euler_26_02_21/",
    # "./data/supermuc/async-euler_26_02_22/",
    # "./data/supermuc/optimize-async-params_26_02_22/",
    # "./data/supermuc/optimize-async-params-2_26_02_22/"
    # "./data/supermuc/optimize-comm-rounds_26_02_22/",
    # "./data/supermuc/optimize-comm-rounds-2_26_02_22/"
    "./data/supermuc/optimize-ultimate-factor_26_02_26/"
    # "./data/supermuc/async-euler-large_26_02_22/"
    # "./data/supermuc/sparse-ruling-set-push-propagation_26_02_19/",
]
# x_param = :sparse_ruling_set_sanders_factor
x_param = :sparse_ruling_set_ultimate_factor
plt = mapping(
    x_param,
    :total_time_mean,
    color=:config,
    marker=:config,
    # linestyle=:sparse_ruling_set_spawn,
    # layout=:graph,
    col=:graph,
    row=:p
) * visual(ScatterLines)
err = mapping(
    x_param,
    # :sparse_ruling_set_sanders_factor,
    # :sparse_ruling_set_round_limit,
    :total_time_min,
    :total_time_max,
    color=:config,
    col=:graph,
    row=:p
) * visual(Rangebars;whiskerwidth=10)

df = vcat(KascadeEval.read.(dirs)...;cols=:union)
filtered_df = df
# filtered_df = @subset(df, :sparse_ruling_set_spawn .== false)
tasks_per_node = foldl(gcd, unique(df.p))

transform!(filtered_df, AsTable(algo_params) => ByRow(t -> to_config_name(;t...)) => :config)

grouped = @by filtered_df [:p, :config, :graph, x_param] begin
    :total_time_mean = mean(:total_time)
    :total_time_min = minimum(:total_time)
    :total_time_max = maximum(:total_time)
end
sort!(grouped, x_param)
# @subset!(grouped, :total_time_mean .<= 13)


figuregrid = draw((plt + err) * data(grouped);
    # axis,
    facet, figure)
display(figuregrid)
save("tmp.pdf", figuregrid)
# save("async-parameters.pdf", figuregrid)
# save("async-on-euler.pdf", figuregrid)
# save("sync-sparse-ruling-set-locality-aware_2026_02_12.pdf", figuregrid)
# save("sparse-ruling-set-base-algorithm_2026_02_19.pdf", figuregrid)

n_loc = 10^6
n_glob(p) = n_loc *p
rulers(f, p) = f * √(n_glob(p)) * p
rounds(f, p) = n_glob(p) / rulers(f, p)
rounds(0.1, 500)
rounds(0.3, 1000)
rounds(0.1, 16000)
