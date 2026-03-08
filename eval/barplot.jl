using DataFrames
using DataFramesMeta
using Statistics
using AlgebraOfGraphics
using CairoMakie
using CategoricalArraysusing DataFrames
using DataFramesMeta
using Statistics
using AlgebraOfGraphics
using CairoMakie
using CategoricalArrays

include("KascadeEval.jl")
import .KascadeEval
include("Config.jl")
df0 = vcat(KascadeEval.read.([
    # "./data/supermuc/async-euler_26_02_21/"
    # "./data/supermuc/sparse-ruling-set-two-level-sync_26_02_23/",
    # "./data/supermuc/sparse-ruling-set-two-level-sync-reverse-list-locality-aware_26_02_24/"
    # "./data/supermuc/local-contraction-filter-perm_26_03_02/",
    # "./data/supermuc/local-contraction_26_03_03/"
    # "./data/supermuc/owner-caching_26_03_03/",
    # "./data/supermuc/impi-none_26_03_03/",
    # "./data/supermuc/impi-plum_26_03_03/",
    # "./data/supermuc/impi-isend_26_03_03/",
    "./data/supermuc/sync-optimized-grid-pointer-doubling_26_03_05/",
    "./data/supermuc/sync-vs-async_26_03_05/",
    "./data/supermuc/sync-three-rounds_26_03_05/"
    # "./data/supermuc/alltoone-gather_26_03_04/"
])...;cols=:union)
# df0.sparse_ruling_set_root_gather_threshold
# df0.i_mpi_adjust_alltoallv

# p = 768 * 2 #* 2 #1536 #* 2 * 2
df = @subset(df0,
    # :graph .== "path(20,permute)" .|| :graph .== "path(20,permute,perm_prob=0.01)",
    # :p .== p,
    # :sparse_ruling_set_use_local_contraction .== true,
    # :sparse_ruling_set_cache_owners .== true,
    # :sparse_ruling_set_ruler_selection .== "sanders",
    # :sparse_ruling_set_sanders_factor .== 1.0,
    # :iteration .== 2,
    # :sparse_ruling_set_grid_comm .== true,
    :sparse_ruling_set_sync .== true,
    # :sparse_ruling_set_ruler_propagation_use_aggregation .== true,
    # :sparse_ruling_set_sync_locality_aware .== true,
)
# df.ruler_propagation = df.ruler_propagation - df.cache_owners_ruler_prop

time_columns = [Symbol(key) for key in keys(Config.timer_value_paths)]
gdf = groupby(df, Not(time_columns, :iteration))
phases = [
    :base_case,
    # :base_case_chase_rulers,
    # :base_case_base_case,
    # :base_case_ruler_propagation,
    # :base_case_post_invert,
    # :base_case_pack_base_case,
    # :base_case_unpack_base_case,
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
    # :cache_owners_ruler_prop,
    :precompute_ruler_permutation,
]
combined = combine(gdf, phases .=> mean, renamecols=false)
combined_long = stack(combined, phases, variable_name=:phase, value_name=:phase_time)
config_keys = [:sparse_ruling_set_grid_comm, :sparse_ruling_set_sparse_ruling_set_rounds]
transform!(combined_long, config_keys => ByRow((x,y) -> "grid=$x,rounds=$y") => :config)
combined_long.graph = categorical(combined_long.graph)



plt = data(combined_long) *
      mapping(:config,:phase_time,
          stack=:phase,
          color=:phase,
          col=:graph,
          row=:p,
      ) *
      visual(BarPlot)

fig = draw(
    plt,
    scales(Color = (; palette = :tab20)),  # sampled for as many categories as needed
    axis = (; xticklabelrotation = π/3),
    figure = (; size = (2000, 2300)),
    facet = (; linkyaxes=false)
)
display(fig)
save("tmp3.pdf", fig)
