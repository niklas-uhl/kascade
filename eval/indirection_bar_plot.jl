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

length(ARGS) == 2 || error("usage: julia indirection_bar_plot.jl <data_dir> <output.pdf>")
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

phases = [:base_case, :chase_rulers, :ruler_propagation]

grouped = @by df [:p, :config] begin
    :total_time = mean(:total_time)
    :base_case = mean(:base_case)
    :chase_rulers = mean(:chase_rulers)
    :ruler_propagation = mean(:ruler_propagation)
end
transform!(grouped, [:total_time, phases...] =>
    ByRow((total, phases...) -> total - sum(phases)) => :rest)

# select three representative core counts spread across the range
ps = sort(unique(grouped.p))
node_size = gcd(ps...)
selected_ps = Set([node_size * 2^k for k in [0, 4, 8] if node_size * 2^k in ps])
isempty(selected_ps) && (selected_ps = Set(ps[[1, max(1, length(ps) ÷ 2), length(ps)]]))
@subset!(grouped, :p .∈ Ref(selected_ps))

config_order = ["Direct", "2DGrid", "TopoAware"]
@assert Set(unique(grouped.config)) == Set(config_order) "unexpected configs: $(unique(grouped.config))"
grouped.config = categorical(grouped.config; levels=config_order, ordered=true)
grouped.p_label = categorical(string.(grouped.p); levels=string.(sort(collect(selected_ps))), ordered=true)

all_phases = [:base_case, :chase_rulers, :ruler_propagation, :rest]
phase_labels = Dict(
    :base_case => "base case",
    :chase_rulers => "ruler chasing",
    :ruler_propagation => "ruler propagation",
    :rest => "other",
)
long = stack(grouped, all_phases, [:p_label, :config], variable_name=:phase, value_name=:phase_time)
transform!(long, :phase => ByRow(p -> phase_labels[Symbol(p)]) => :phase)

phase_order = ["ruler chasing", "ruler propagation", "base case", "other"]
long.phase = categorical(long.phase; levels=phase_order, ordered=true)

plt = data(long) *
    mapping(:p_label, :phase_time; color=:phase, stack=:phase, col=:config) *
    visual(BarPlot)

figuregrid = draw(plt;
    axis=(; xlabel="# cores", ylabel=L"\mathrm{Time}\ /s"),
    facet=(; linkyaxes=:row),
    figure=(; size=(900, 500)))
save(output_file, figuregrid)
