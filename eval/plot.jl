using DataFrames
using DataFramesMeta
using Statistics
using AlgebraOfGraphics
using CairoMakie

include("KascadeEval.jl")
import .KascadeEval

dirs = [
    "./data/horeka/pointer-doubling_26_01_30/",
    "./data/horeka/async-pointer-doubling_26_01_30/"
]

# -------------------------
# Data loading & transform
# -------------------------
algo_params = [
    :algorithm,
    :async_caching,
    :pointer_doubling_aggregation_level,
    :pointer_doubling_use_local_preprocessing,
    :sparse_ruling_set_sync,
    :sparse_ruling_set_spawn,
    :sparse_ruling_set_dehne_factor,
    :eulertour_algorithm,
    :commit
]
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
        if kwargs[:sparse_ruling_set_sync] == true
            name *= "-sync"
        else
            name *= "-async"
        end
        if kwargs[:sparse_ruling_set_spawn] == true
            name *= "-spawn"
        end
        name *= " (ruler_factor=$(kwargs[:sparse_ruling_set_dehne_factor]))"
    end
    return name
end

df = vcat(KascadeEval.read.(dirs)...)
transform!(df, AsTable(algo_params) => ByRow(t -> to_config_name(;t...)) => :config)

grouped = @by df [:p, :config, :graph] begin
    :total_time_mean = mean(:total_time)
end

# -------------------------
# Plotting
# -------------------------

plt = data(grouped) *
      mapping(:p, :total_time_mean,
          color=:config,
          marker=:config,
          layout=:graph) * 
      visual(ScatterLines)

axis = (; xscale=log2)
facet = (; linkyaxes=:minimal)
figure = (; size=(1000, 1200))
figuregrid = draw(plt; axis, facet, figure)
display(figuregrid)
