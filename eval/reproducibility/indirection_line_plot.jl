using DataFrames
using DataFramesMeta
using Statistics
using CairoMakie
using AlgebraOfGraphics
using LaTeXStrings
using CategoricalArrays
using ArgParse

include("../KascadeEval.jl")
import .KascadeEval
include("../Config.jl")
import .Config

function parse_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "data_dir"
            help = "path to sparse-ruling-set-indirection experiment output"
            required = true
        "--output", "-o"
            help = "output PDF path"
            default = "indirection_scatter_plot.pdf"
    end
    return ArgParse.parse_args(s)
end
args = parse_args()
data_dir    = args["data_dir"]
output_file = args["output"]

isdir(data_dir) || error("data directory not found: $data_dir")

df = KascadeEval.read(data_dir)

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

node_size = gcd(unique(grouped.p)...)
grouped.p_exp = ceil.(Int, log2.(grouped.p ./ node_size))
ks = sort(unique(grouped.p_exp))
xtick_positions = ks
xtick_labels = [L"{%$node_size} \times 2^{%$k} = %$(node_size * 2^k)" for k in ks]

plt = mapping(
    :p_exp,
    :total_time_mean,
    color=:config,
    marker=:config
) * visual(ScatterLines)
err = mapping(
    :p_exp,
    :total_time_min,
    :total_time_max,
    color=:config,
) * visual(Rangebars; whiskerwidth=10)

figuregrid = draw((plt + err) * data(grouped),
    scales(Color=(; palette=[:green, "darkorange", :purple]),
           Marker=(; palette=[:circle, :rect, :utriangle]));
    axis=(;
          xticks=(xtick_positions, xtick_labels),
          xticklabelrotation=π/4,
          xlabel="# cores",
          ylabel=L"\mathrm{Total\ time}\ /s",
    ),
    figure=(; size=(600, 500)))
save(output_file, figuregrid)
