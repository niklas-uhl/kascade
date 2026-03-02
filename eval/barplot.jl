using DataFrames
using DataFramesMeta
using Statistics
using AlgebraOfGraphics
using CairoMakie

include("KascadeEval.jl")
import .KascadeEval
include("Config.jl")
df0 = vcat(KascadeEval.read.([
    # "./data/supermuc/async-euler_26_02_21/"
    # "./data/supermuc/sparse-ruling-set-two-level-sync_26_02_23/",
    # "./data/supermuc/sparse-ruling-set-two-level-sync-reverse-list-locality-aware_26_02_24/"
    "./data/supermuc/local-contraction-filter-perm_26_03_02/"
])...;cols=:union)

p = 768 #* 2 #1536 #* 2 * 2
df = @subset(df0,
    # :graph .== "path(20)",
    :p .== p,
    # :sparse_ruling_set_ruler_selection .== "sanders",
    # :sparse_ruling_set_sanders_factor .== 1.0,
    # :iteration .== 2,
    # :sparse_ruling_set_grid_comm .== true,
    # :sparse_ruling_set_ruler_propagation_use_aggregation .== true,
    # :sparse_ruling_set_sync_locality_aware .== true,
)


time_columns = [Symbol(key) for key in keys(Config.timer_value_paths)]
gdf = groupby(df, Not(time_columns, :iteration))
phases = [
    :base_case,
    :chase_rulers,
    :init_node_type,
    :find_rulers,
    :invert_list,
    :ruler_propagation,
    :pack_base_case,
    :unpack_base_case,
    :find_leaves,
    :post_invert,
    :fixup_unreached,
    :local_contraction,
    :local_uncontraction,
    :precompute_ruler_permutation,
]
combined = combine(gdf, phases .=> mean, renamecols=false)
combined_long = stack(combined, phases, variable_name=:phase, value_name=:phase_time)
config_keys = [:sparse_ruling_set_grid_comm, :sparse_ruling_set_use_local_contraction]
transform!(combined_long, config_keys => ByRow((x,y) -> "grid=$x,contract=$y") => :config)

plt = data(combined_long) * mapping(:config,:phase_time,stack=:phase,color=:phase,col=:graph) * visual(BarPlot)
axis = (;xticklabelrotation=π/3)
figure = (; size=(2000, 1200))

fig = draw(
    plt,
    scales(Color = (; palette = :tab20)),  # sampled for as many categories as needed
    axis = (; xticklabelrotation = π/3),
    figure = (; size = (2000, 1200)),
)
display(fig)
save("tmp3.pdf", fig)
