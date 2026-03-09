using DataFrames
using DataFramesMeta
using Statistics
using AlgebraOfGraphics
using CairoMakie

include("KascadeEval.jl")
import .KascadeEval
include("Config.jl")
import .Config

plt = mapping(
    :p,
    :total_time_mean,
    color=:config,
    layout=:graph,
    marker=:config
) * visual(ScatterLines)
err = mapping(
    :p,
    :total_time_min,
    :total_time_max,
    color=:config,
    layout=:graph
) * visual(Rangebars;whiskerwidth=10)

dirs = [
    "./data/supermuc/sparse-ruling-set-indirection-no-intel-tuning_26_03_08/",
]
df = vcat(KascadeEval.read.(dirs)...;cols=:union)

transform!(df, AsTable(:) => ByRow(t -> Config.to_config_name(;t...)) => :config)

additional_group_keys = []

grouped = @by df [:p, :config, :graph, additional_group_keys...] begin
    :total_time_mean = mean(:total_time)
    :total_time_min = minimum(:total_time)
    :total_time_max = maximum(:total_time)
end

grouped.p ./= 48


figuregrid = draw((plt + err) * data(grouped);
    axis=(; xscale=log2),
    facet=(;linkyaxes=:none),
    figure=(; size=(2000, 1000)))
display(figuregrid)
# save("tmp.pdf", figuregrid)


