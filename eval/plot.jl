using DataFrames
using DataFramesMeta
using Statistics
using AlgebraOfGraphics
using CairoMakie

include("KascadeEval.jl")

dirs = [
    "./data/horeka/pointer-doubling_26_01_30/",
    "./data/horeka/async-pointer-doubling_26_01_30/"
]

# -------------------------
# Data loading & transform
# -------------------------

function to_config_name(algorithm, caching)
    name = "$(algorithm)"
    if algorithm == "AsyncPointerDoubling" && caching
        name *= "+cache"
    end
    return name
end

df = vcat(KascadeEval.read.(dirs)...)
@transform!(df, :config = to_config_name.(:algorithm, :async_caching))

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
          row=:graph) * 
      visual(ScatterLines)

axis = (; xscale=log2)
facet = (;linkyaxes = :minimal)

figuregrid = draw(plt; axis, facet)
display(figuregrid)
