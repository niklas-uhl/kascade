using DataFrames
using DataFramesMeta
using Statistics
using AlgebraOfGraphics
using CairoMakie

include("KascadeEval.jl")
import .KascadeEval
include("Config.jl")
df0 = KascadeEval.read("./data/supermuc/sparse-ruling-set-two-level-sync_26_02_23/")
p = 1536 #* 2 * 2
df = @subset(df0,
    # :graph .== "path(20,permute)",
    :p .== p,
    :sparse_ruling_set_ruler_selection .== "sanders",
    :sparse_ruling_set_sanders_factor .== 1.0,
    # :iteration .== 2,
    # :sparse_ruling_set_grid_comm .== true,
    :sparse_ruling_set_ruler_propagation_use_aggregation .== true,
    :sparse_ruling_set_sync_locality_aware .== true,
)
time_columns = [Symbol(key) for key in keys(Config.timer_value_paths)]
gdf = groupby(df, Not(time_columns, :iteration))
phases = [:base_case, :chase_rulers, :invert_list, :ruler_propagation, :pack_base_case, :unpack_base_case]
combined = combine(gdf, phases .=> mean, renamecols=false)
combined_long = stack(combined, phases, variable_name=:phase, value_name=:phase_time)
plt = data(combined_long) * mapping(:sparse_ruling_set_grid_comm,:phase_time,stack=:phase,color=:phase,col=:graph) * visual(BarPlot)
# axis = (;linkyaxes=:minimal)
fig = draw(plt)
display(fig)





