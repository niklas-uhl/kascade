using DataFrames
using DataFramesMeta
using Statistics
using AlgebraOfGraphics
using CairoMakie

include("KascadeEval.jl")
import .KascadeEval

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
    :sparse_ruling_set_round_limit,
    :sparse_ruling_set_ruler_propagation_mode,
    :sparse_ruling_set_ruler_propagation_use_aggregation,
    :sparse_ruling_set_dehne_factor,
    :sparse_ruling_set_ultimate_factor,
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
        else
            name *= "-async"
        end
        if kwargs[:sparse_ruling_set_grid_comm] == true
            name *= "-grid-comm"
        end
        if kwargs[:sparse_ruling_set_spawn] == true
            name *= "-spawn"
        end
        name *= " (ruler_selection=$(kwargs[:sparse_ruling_set_ruler_selection])"
        if kwargs[:sparse_ruling_set_ruler_selection] == "dehne"
            name *= ", dehne_factor=$(kwargs[:sparse_ruling_set_dehne_factor])"
        elseif kwargs[:sparse_ruling_set_ruler_selection] == "sanders"
            name *= ", sanders_factor=$(kwargs[:sparse_ruling_set_sanders_factor])"
        elseif kwargs[:sparse_ruling_set_ruler_selection] == "limit-rounds"
            name *= ", round_limit=$(kwargs[:sparse_ruling_set_round_limit])"
        elseif kwargs[:sparse_ruling_set_ruler_selection] == "ultimate"
            name *= ", round_limit=$(kwargs[:sparse_ruling_set_ultimate_factor])"
        end
        name *= ")"
        if kwargs[:sparse_ruling_set_ruler_propagation_mode] != "pull"
            name *= " (ruler_propagation_mode=$(kwargs[:sparse_ruling_set_ruler_propagation_mode]))"
        end
        if kwargs[:sparse_ruling_set_ruler_propagation_use_aggregation] == true
            name *= " (ruler_prop_agg)"
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

axis = (; xscale=log2,
        # limits=(nothing,(0,nothing)),
        # yscale=log10
)
facet = (; linkyaxes=:none)
figure = (; size=(2000, 1000))

dirs = [
    # "./data/supermuc/sync-vs-async_26_03_05/",
    # "./data/supermuc/async-no-spawn_26_03_05/",
    "./data/supermuc/sync-optimized-grid-pointer-doubling_26_03_05/"
    # "./data/supermuc/async-no-spawn-more-rulers_26_03_05/"
    # "./data/supermuc/sync-optimized_26_03_05/"
]
x_param = :p
plt = mapping(
    x_param,
    :total_time_mean,
    color=:config,
    layout=:graph,
    marker=:config
    # col=:graph
    # marker=:config,
    # col=:sparse_ruling_set_grid_comm,
    # layout=:graph,
    # row=:graph
    # linestyle=:sparse_ruling_set_ruler_selection
    # col=:graph,
    # row=:p
) * visual(ScatterLines)
err = mapping(
    x_param,
    # :sparse_ruling_set_sanders_factor,
    # :sparse_ruling_set_round_limit,
    :total_time_min,
    :total_time_max,
    color=:config,
    # col=:sparse_ruling_set_grid_comm,
    layout=:graph
    # col=:graph,
    # row=:p
) * visual(Rangebars;whiskerwidth=10)

df = vcat(KascadeEval.read.(dirs)...;cols=:union)
# @subset!(df, :sparse_ruling_set_sync .== true)

transform!(df, AsTable(algo_params) => ByRow(t -> to_config_name(;t...)) => :config)

additional_group_keys = [] #:sparse_ruling_set_ruler_selection, :sparse_ruling_set_grid_comm]

grouped = @by df [:p, :config, :graph, additional_group_keys...] begin
    :total_time_mean = mean(:total_time)
    :total_time_min = minimum(:total_time)
    :total_time_max = maximum(:total_time)
end
grouped.graph = categorical(grouped.graph)
grouped.p ./= 48


figuregrid = draw((plt + err) * data(grouped);axis,facet, figure)
display(figuregrid)
save("tmp.pdf", figuregrid)
# save("async-on-euler.pdf", figuregrid)
# save("sync-sparse-ruling-set-locality-aware_2026_02_12.pdf", figuregrid)
# save("sparse-ruling-set-base-algorithm_2026_02_19.pdf", figuregrid)










