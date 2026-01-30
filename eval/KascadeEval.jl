module KascadeEval

using DataFramesMeta

include("Config.jl")
include("LogProcessing.jl")

function cleanup(df)
    function expand_timers(timer_dict)
        (; (Symbol(key) => Float64(LogProcessing.get_json_path(timer_dict, path)[1])
            for (key, path) in Config.timer_value_paths)...)
    end
    if isempty(df)
        return df
    end
    transform!(df, :time => ByRow(expand_timers) => AsTable)
    allowed_kagen_args = Set(["type", "n", "m", "permute"])
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
                result[Symbol(key)] = parse_value(val)
            end
        end
        for key in allowed_kagen_args
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
    function to_graph_string(type, n, m)
        result = "$type("
        if n !== missing
            result *= "$(Int(log2(n)))"
        end
        if m !== missing
            result *= ",$(Int(log2(m)))"
        end
        result *= ")"
        return result
    end
    transform!(df, [:type, :n, :m] => ByRow(to_graph_string) => :graph)
    return select!(df, Not([:kagen_option_string, :time]))
end

function read(dir)
    df = LogProcessing.read_logs_from_directory(dir, Config.value_paths)
    return cleanup(df)
end
end
