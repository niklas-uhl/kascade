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
        "srs_dir"
            help = "path to sparse-ruling-set experiment output"
            required = true
        "pd_dir"
            help = "path to pointer-doubling experiment output"
            required = true
        "--output", "-o"
            help = "output PDF path"
            default = "scalability_plot.pdf"
    end
    return ArgParse.parse_args(s)
end
args = parse_args()
srs_dir     = args["srs_dir"]
pd_dir      = args["pd_dir"]
output_file = args["output"]

isdir(srs_dir) || error("data directory not found: $srs_dir")
isdir(pd_dir)  || error("data directory not found: $pd_dir")

df = vcat(
    KascadeEval.read(srs_dir),
    KascadeEval.read(pd_dir);
    cols=:union
)

function scalability_config_name(algorithm, srs_grid_comm, pd_grid_comm)
    if algorithm == "SparseRulingSet"
        return coalesce(srs_grid_comm, false) ? "SRS+Ind" : "SRS"
    elseif algorithm == "PointerDoubling"
        return coalesce(pd_grid_comm, false) ? "PD+Ind" : "PD"
    else
        error("unexpected algorithm: $algorithm")
    end
end
transform!(df, [:algorithm, :sparse_ruling_set_grid_comm, :pointer_doubling_grid_comm] =>
    ByRow(scalability_config_name) => :config)

graph_rename = Dict(
    "path(16,permute)" => L"\mathrm{List}(2^{16},\ \gamma=1.0)",
    "path(18,permute)" => L"\mathrm{List}(2^{18},\ \gamma=1.0)",
    "path(20,permute)" => L"\mathrm{List}(2^{20},\ \gamma=1.0)",
    "path(22,permute)" => L"\mathrm{List}(2^{22},\ \gamma=1.0)",
    "gnm(19,22)"       => L"\mathrm{GNM}(2^{19},\ 2^{22})\ \mathrm{ET}",
    "rgg2d(19,22)"     => L"\mathrm{RGG2D}(2^{19},\ 2^{22})\ \mathrm{ET}",
)
graph_order = [
    graph_rename["path(16,permute)"], graph_rename["path(18,permute)"],
    graph_rename["path(20,permute)"], graph_rename["path(22,permute)"],
    graph_rename["gnm(19,22)"],       graph_rename["rgg2d(19,22)"],
]
transform!(df, :graph => ByRow(g -> get(graph_rename, g, g)) => :graph)

sort!(df, :p)

grouped = @by df [:p, :config, :graph] begin
    :nrows = length(:iteration)
    :total_time_mean = mean(:total_time)
    :total_time_min = minimum(:total_time)
    :total_time_max = maximum(:total_time)
end

config_order = ["PD", "PD+Ind", "SRS", "SRS+Ind"]
@assert Set(unique(grouped.config)) == Set(config_order) "unexpected configs: $(unique(grouped.config))"
grouped.config = categorical(grouped.config; levels=config_order, ordered=true)

@assert length(unique(grouped.graph)) == 6 "unexpected graphs: $(unique(grouped.graph))"
grouped.graph = categorical(grouped.graph; levels=graph_order, ordered=true)

node_size = gcd(unique(grouped.p)...)
grouped.p_exp = ceil.(Int, log2.(grouped.p ./ node_size))
ks = sort(unique(grouped.p_exp))
xtick_positions = ks
xtick_labels = [L"{%$node_size} \times 2^{%$k} = %$(node_size * 2^k)" for k in ks]

plt = mapping(
    :p_exp,
    :total_time_mean,
    color=:config,
    layout=:graph,
    marker=:config
) * visual(ScatterLines)
err = mapping(
    :p_exp,
    :total_time_min,
    :total_time_max,
    color=:config,
    layout=:graph
) * visual(Rangebars; whiskerwidth=10)

figuregrid = draw((plt + err) * data(grouped),
    scales(Color=(; palette=[:green, "darkorange", :purple, "hotpink"]),
           Marker=(; palette=[:circle, :rect, :utriangle, :diamond]));
    axis=(;
          xticks=(xtick_positions, xtick_labels),
          xticklabelrotation=π/4,
          xlabel="# cores",
          ylabel="Total time /s"
    ),
    facet=(; linkyaxes=:none),
    figure=(; size=(2000, 1000)))
save(output_file, figuregrid)
