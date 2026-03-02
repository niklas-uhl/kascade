module KascadeEval

using DataFramesMeta

include("Config.jl")
include("LogProcessing.jl")

function cleanup(df)
    function expand_timers(timer_dict)
        path_to_val(path) = begin
            val = LogProcessing.get_json_path(timer_dict, path)
            if val === nothing
                return missing
            else
                return Float64(val[1])
            end
        end
        (; (Symbol(key) => path_to_val(path)
            for (key, path) in Config.timer_value_paths)...)
    end
    if isempty(df)
        return df
    end
    transform!(df, :time => ByRow(expand_timers) => AsTable)
    allowed_kagen_args = Set(["type", "n", "m", "p", "permute", "gamma", "permutation_prob"])
    args_remap = Dict("p" => "prob")
    function expand_graph(kagen_option_string::String)
        function parse_value(val)
            try
                return parse(Int, val)
            catch
            end
            try
                return parse(Float64, val)
            catch
            end
            return val
        end
        pairs = split(kagen_option_string, ';')
        result = Dict{Symbol, Any}()
        result[:permute] = false # TODO: remove this
        for pair in pairs
            # treat as bool if pair contains no '='
            if !occursin('=', pair)
                key = pair
                val = true
            else
                key, val = split(pair, '=')
            end
            if key in allowed_kagen_args
                if haskey(args_remap, key)
                    key = args_remap[key]
                end
                result[Symbol(key)] = parse_value(val)
            end
        end
        for key in allowed_kagen_args
            if haskey(args_remap, key)
                key = args_remap[key]
            end
            if Symbol(key) ∉ keys(result)
                result[Symbol(key)] = missing
            end
        end
        return (; result...)
    end
    transform!(df, :kagen_option_string => ByRow(expand_graph) => AsTable)
    df.n = trunc.(Union{Int, Missing}, df.n ./ df.p)
    df.m = trunc.(Union{Int, Missing}, df.m ./ df.p)
    replace!(df.type, "gnm-undirected" => "gnm")
    @subset!(df, :iteration .>= Config.first_iteration)
    sort!(df, :p)
    function to_graph_string(;kwargs...)
        type = kwargs[:type]
        n = kwargs[:n]
        m = kwargs[:m]
        result = "$type("
        if n !== missing
            result *= "$(Int(log2(n)))"
        end
        if m !== missing
            result *= ",$(Int(log2(m)))"
        end
        if kwargs[:permute] == true
            result *= ",permute"
        end
        if kwargs[:gamma] !== missing
            result *= ",gamma=$(kwargs[:gamma])"
        end
        if kwargs[:prob] !== missing
            result *= ",prob=$(kwargs[:prob])"
        end
        if kwargs[:permutation_prob] !== missing
            result *= ",perm_prob=$(kwargs[:permutation_prob])"
        end
        result *= ")"
        return result
    end
    transform!(df, AsTable(:) => ByRow(t -> to_graph_string(;t...)) => :graph)
    return select!(df, Not([:kagen_option_string, :time]))
end

function read(dir)
    df = LogProcessing.read_logs_from_directory(dir, Config.value_paths)
    return cleanup(df)
end
end
