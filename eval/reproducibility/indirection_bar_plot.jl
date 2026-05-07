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
            default = "indirection_bar_plot.pdf"
        "--ps"
            help = "comma-separated list of core counts to show (default: auto-select 3)"
            default = nothing
    end
    return ArgParse.parse_args(s)
end
args = parse_args()
data_dir    = args["data_dir"]
output_file = args["output"]
requested_ps = isnothing(args["ps"]) ? nothing : Set(parse.(Int, split(args["ps"], ',')))

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

phases = [:base_case, :chase_rulers, :ruler_propagation]

grouped = @by df [:p, :config] begin
    :total_time = mean(:total_time)
    :base_case = mean(:base_case)
    :chase_rulers = mean(:chase_rulers)
    :ruler_propagation = mean(:ruler_propagation)
end
transform!(grouped, [:total_time, phases...] =>
    ByRow((total, phases...) -> total - sum(phases)) => :rest)

ps = sort(unique(grouped.p))
node_size = gcd(ps...)
selected_ps = isnothing(requested_ps) ? Set(ps) : requested_ps
@subset!(grouped, :p .∈ Ref(selected_ps))

config_order = ["Direct", "2DGrid", "TopoAware"]
@assert Set(unique(grouped.config)) == Set(config_order) "unexpected configs: $(unique(grouped.config))"
grouped.config = categorical(grouped.config; levels=config_order, ordered=true)
p_label(p) = L"{%$node_size} \times 2^{%$(ceil(Int, log2(p / node_size)))} = %$p"
sorted_selected = sort(collect(selected_ps))
grouped.p_label = categorical(p_label.(grouped.p); levels=p_label.(sorted_selected), ordered=true)

all_phases = [:base_case, :chase_rulers, :ruler_propagation, :rest]
phase_labels = Dict(
    :base_case => "base case",
    :chase_rulers => "ruler chasing",
    :ruler_propagation => "ruler propagation",
    :rest => "other",
)
long = stack(grouped, all_phases, [:p_label, :config], variable_name=:phase, value_name=:phase_time)
transform!(long, :phase => ByRow(p -> phase_labels[Symbol(p)]) => :phase)

phase_order = ["base case", "ruler chasing", "ruler propagation", "other"]
long.phase = categorical(long.phase; levels=phase_order, ordered=true)

plt = data(long) *
    mapping(:p_label, :phase_time; color=:phase, stack=:phase, col=:config) *
    visual(BarPlot)

figuregrid = draw(plt,
    scales(Color=(; palette=[:green, "darkorange", :purple, "hotpink"]));
    axis=(; xlabel="# cores", ylabel=L"\mathrm{Time}\ /s", xticklabelrotation=π/4),
    facet=(; linkyaxes=:rowwise),
    figure=(; size=(900, 500)))
save(output_file, figuregrid)
