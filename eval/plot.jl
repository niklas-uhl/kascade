using DataFrames
using DataFramesMeta
using Statistics
using CairoMakie
using AlgebraOfGraphics
using LaTeXStrings
using CategoricalArrays
using CSV

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

dirs = [
    "./data/supermuc/sparse-ruling-set-indirection-no-intel-tuning_26_03_08/",
    "./data/supermuc/sparse-ruling-set-indirection-intel-plum_26_03_08/",
    "./data/supermuc/sparse-ruling-set-indirection-intel-isend_26_03_08/",
]
df = vcat(KascadeEval.read.(dirs)...;cols=:union)

transform!(df, AsTable(:) => ByRow(t -> Config.to_config_name(;t...)) => :config)
df.config = categorical(df.config)
filtered = @subset(df,
    # filter conditions
)

additional_group_keys = []

grouped = @by filtered [:p, :config, :graph, additional_group_keys...] begin
    :total_time_mean = mean(:total_time)
    :total_time_min = minimum(:total_time)
    :total_time_max = maximum(:total_time)
end

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
function write_plot_data(df; output_path="./", output_prefix="")
    mkpath(output_path)
    df_out = grouped
    df_out.graph_id = levelcode.(df_out.graph)
    df_out.config_id = levelcode.(df_out.config)
    df_out = df_out[!, [:p_exp, :graph_id, :config_id, :total_time_mean, :total_time_min, :total_time_max]]
    groups = groupby(df_out, [:graph_id, :config_id])
    for (key,g) in pairs(groups)
        CSV.write(joinpath(output_path, output_prefix * "data-in$(key[1])-c$(key[2]).csv"), g)
    end
    graph_map = DataFrame(
        graph = levels(df.graph),
        graph_id = levelcode.(categorical(levels(df.graph); levels=levels(df.graph)))
    )
    config_map = DataFrame(
        config = levels(df.config),
        config_id = levelcode.(categorical(levels(df.config); levels=levels(df.config)))
    )
    CSV.write(joinpath(output_path, output_prefix * "graph_map.csv"), graph_map)
    CSV.write(joinpath(output_path, output_prefix * "config_map.csv"), config_map)
end

write_plot_data(grouped, output_path="./../../kascade-paper/plots/indirection/", output_prefix="indirection-")

