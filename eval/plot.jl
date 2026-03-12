using DataFrames
using DataFramesMeta
using Statistics
using CairoMakie
using AlgebraOfGraphics
using LaTeXStrings
using CategoricalArrays
using CSV
using Glob

include("KascadeEval.jl")
import .KascadeEval
include("Config.jl")
import .Config

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
) * visual(Rangebars;whiskerwidth=10)

# dirs = [
#     "/Users/niklasuhl/Downloads/submission_data_all/sparse-ruling-set-locality_26_03_11/"
# ]
# dirs = Glob.glob("*", "/Users/niklasuhl/Downloads/submission_data_all/scalability/")
dirs = Glob.glob("*", "/Users/niklasuhl/Desktop/very_final_data/scalability/euler/")
df = vcat(KascadeEval.read.(dirs)...;cols=:union)

transform!(df, AsTable(:) => ByRow(t -> Config.to_config_name(;t...)) => :config)
df.config = categorical(df.config)


filters_indirection = [
    quote
        @subset(df,
            :algorithm .== "SparseRulingSet",
             :i_mpi_adjust_alltoallv .== 1,
            :sparse_ruling_set_use_local_contraction .== true,
            :permute,
            :permutation_prob .=== missing
        )
    end,
]

filters_locality = [
    quote
        @subset(df,
            :n .== 2^20,
            :sparse_ruling_set_grid_comm .== false,
        )
    end,
]

filters_scalability_euler = [
    quote
        @subset(df,
            :algorithm .== "SparseRulingSet",
            :i_mpi_adjust_alltoallv .== "",
            :sparse_ruling_set_use_local_contraction .== true,
            :sparse_ruling_set_sync_locality_aware,
            :sparse_ruling_set_grid_comm,
            :sparse_ruling_set_grid_communicator_mode .== "topology-aware",
        )
    end,
    quote
        @subset(df,
            :algorithm .== "SparseRulingSet",
            :i_mpi_adjust_alltoallv .== "",
            :sparse_ruling_set_use_local_contraction .== true,
            :sparse_ruling_set_sync_locality_aware,
            :sparse_ruling_set_grid_comm .== false,
        )
    end,
    quote
        @subset(df,
            :algorithm .== "PointerDoubling",
        )
    end
]
filters = filters_scalability_euler


filtered_dfs = [eval(f) for f in filters]
filtered = vcat(filtered_dfs...)
sort!(filtered, :p)

additional_group_keys = [] #[:class, :instance]

grouped = @by filtered [:p, :config, :graph, additional_group_keys...] begin
    :nrows = length(:iteration)
    :total_time_mean = mean(:total_time)
    :total_time_min = minimum(:total_time)
    :total_time_max = maximum(:total_time)
end

unique(grouped.nrows)
grouped.p_exp = ceil.(Int, log2.(grouped.p ./ 48))

ks = 0:maximum(grouped.p_exp)
xtick_positions = ks
xtick_labels = [L"48 \times 2^{%$k} = %$(48 * 2^k)" for k in ks]


figuregrid = draw((plt + err) * data(grouped);
    axis=(;
          xticks = (xtick_positions, xtick_labels),
          xticklabelrotation = π/4,
          xlabel = "# cores",
          ylabel = "Total time /s"
    ),
    facet=(; linkyaxes=:none),
    figure=(; size=(2000, 1000)))
display(figuregrid)
# save("tmp.pdf", figuregrid)


# write CSV data that we can ingest in PGFPlots
function write_plot_data(result_df; output_path="./", output_prefix="")
    mkpath(output_path)
    result_df = copy(result_df)
    droplevels!(result_df.graph)
    droplevels!(result_df.config)

    result_df.graph_id = levelcode.(result_df.graph)
    result_df.config_id = levelcode.(result_df.config)
    df_out = result_df[!, [:p_exp, :graph_id, :config_id, :total_time_mean, :total_time_min, :total_time_max]]
    groups = groupby(df_out, [:graph_id, :config_id])
    for (key,g) in pairs(groups)
        CSV.write(joinpath(output_path, output_prefix * "data-in$(key[1])-c$(key[2]).csv"), g)
    end
    graph_map = DataFrame(
        graph = levels(result_df.graph),
        graph_id = levelcode.(categorical(levels(result_df.graph); levels=levels(result_df.graph)))
    )
    config_map = DataFrame(
        config = levels(result_df.config),
        config_id = levelcode.(categorical(levels(result_df.config); levels=levels(result_df.config)))
    )
    print(config_map)
    CSV.write(joinpath(output_path, output_prefix * "graph_map.csv"), graph_map)
    CSV.write(joinpath(output_path, output_prefix * "config_map.csv"), config_map)
end
write_plot_data(grouped, output_path="./../../kascade-paper/plots/scalability/euler", output_prefix="scalability-euler-")
