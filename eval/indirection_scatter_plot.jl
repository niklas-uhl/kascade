using DataFrames
using DataFramesMeta
using Statistics
using CairoMakie
using AlgebraOfGraphics
using LaTeXStrings
using CategoricalArrays

include("KascadeEval.jl")
import .KascadeEval
include("Config.jl")
import .Config

length(ARGS) == 2 || error("usage: julia indirection_scatter_plot.jl <data_dir> <output.pdf>")
data_dir, output_file = ARGS

isdir(joinpath(data_dir, "sparse-ruling-set-indirection")) ||
    error("data directory not found: $(joinpath(data_dir, "sparse-ruling-set-indirection"))")

df = KascadeEval.read(joinpath(data_dir, "sparse-ruling-set-indirection"))

function indirection_config_name(grid_comm, grid_mode)
    coalesce(grid_comm, false) || return "Direct"
    grid_mode == "balanced" && return "2DGrid"
    return "TopoAware"
end
transform!(df, [:sparse_ruling_set_grid_comm, :sparse_ruling_set_grid_communicator_mode] =>
    ByRow(indirection_config_name) => :config)

sort!(df, :p)

grouped = @by df [:p, :config] begin
    :nrows = length(:iteration)
    :total_time_mean = mean(:total_time)
    :total_time_min = minimum(:total_time)
    :total_time_max = maximum(:total_time)
end

config_order = ["Direct", "2DGrid", "TopoAware"]
@assert Set(unique(grouped.config)) == Set(config_order) "unexpected configs: $(unique(grouped.config))"
grouped.config = categorical(grouped.config; levels=config_order, ordered=true)

plt = mapping(
    :p,
    :total_time_mean,
    color=:config,
    marker=:config,
    layout=:graph
) * visual(ScatterLines)
err = mapping(
    :p,
    :total_time_min,
    :total_time_max,
    color=:config,
    layout=:graph
) * visual(Rangebars; whiskerwidth=10)

figuregrid = draw((plt + err) * data(grouped);
    axis=(;
          xscale=log2,
          xlabel="# cores",
          ylabel=L"\mathrm{Total\ time}\ /s",
    ),
    facet=(; linkyaxes=:none),
    figure=(; size=(1200, 500)))
save(output_file, figuregrid)
