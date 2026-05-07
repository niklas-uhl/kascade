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
            help = "path to sparse-ruling-set-locality experiment output"
            required = true
        "--output", "-o"
            help = "output PDF path"
            default = "locality_plot.pdf"
    end
    return ArgParse.parse_args(s)
end
args = parse_args()
data_dir    = args["data_dir"]
output_file = args["output"]

isdir(data_dir) || error("data directory not found: $data_dir")

plt = mapping(
    :p_exp,
    :total_time_mean,
    color=:config,
    col=:graph,
    marker=:config
) * visual(ScatterLines)
err = mapping(
    :p_exp,
    :total_time_min,
    :total_time_max,
    color=:config,
    col=:graph
) * visual(Rangebars;whiskerwidth=10)

df = vcat(KascadeEval.read.([data_dir])...;cols=:union)

transform!(df, AsTable(:) => ByRow(t -> Config.to_config_name(;t...)) => :config)
df.config = categorical(df.config)

graph_rename = Dict(
    "path(20)"                        => L"\mathrm{List}(2^{20},\ \gamma = 0.0)",
    "path(20,permute,perm_prob=0.01)" => L"\mathrm{List}(2^{20},\ \gamma = 0.01)",
    "path(20,permute,perm_prob=0.1)"  => L"\mathrm{List}(2^{20},\ \gamma = 0.1)",
    "path(20,permute)"                => L"\mathrm{List}(2^{20},\ \gamma = 1.0)",
)
graph_order = [graph_rename["path(20)"], graph_rename["path(20,permute,perm_prob=0.01)"],
               graph_rename["path(20,permute,perm_prob=0.1)"], graph_rename["path(20,permute)"]]
transform!(df, :graph => ByRow(g -> get(graph_rename, g, g)) => :graph)

filtered = df
sort!(filtered, :p)

additional_group_keys = [] #[:class, :instance]

grouped = @by filtered [:p, :config, :graph, additional_group_keys...] begin
    :nrows = length(:iteration)
    :files = join(unique(:source), ",")
    :total_time_mean = mean(:total_time)
    :total_time_min = minimum(:total_time)
    :total_time_max = maximum(:total_time)
    :locality_aware = any(:sparse_ruling_set_sync_locality_aware .=== true)
    :local_contraction = any(:sparse_ruling_set_use_local_contraction .=== true)
end

function locality_config_name(locality_aware, local_contraction)
    local_contraction && return "local contraction"
    locality_aware && return "local chasing"
    return "plain"
end
transform!(grouped, [:locality_aware, :local_contraction] => ByRow(locality_config_name) => :config)
@assert length(unique(grouped.config)) == 3 "unexpected configs: $(unique(grouped.config))"
config_order = ["plain", "local chasing", "local contraction"]
grouped.config = categorical(grouped.config; levels=config_order, ordered=true)

@assert length(unique(grouped.graph)) == 4 "unexpected graphs: $(unique(grouped.graph))"
grouped.graph = categorical(grouped.graph; levels=graph_order, ordered=true)

node_size = gcd(unique(grouped.p)...)
grouped.p_exp = ceil.(Int, log2.(grouped.p ./ node_size))

ks = 0:maximum(grouped.p_exp)
xtick_positions = ks
xtick_labels = [L"{%$node_size} \times 2^{%$k} = %$(node_size * 2^k)" for k in ks]


figuregrid = draw((plt + err) * data(grouped),
    scales(Color=(; palette=[:green, "darkorange", :purple]), Marker=(; palette=[:circle, :rect, :utriangle]));
    axis=(;
          xticks = (xtick_positions, xtick_labels),
          xticklabelrotation = π/4,
          xlabel = "# cores",
          ylabel = "Total time /s"
    ),
    facet=(; linkyaxes=:none),
    figure=(; size=(1500, 500)))
save(output_file, figuregrid)
