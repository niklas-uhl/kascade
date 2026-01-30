module LogProcessing

using DataFrames
using JSON
using Glob
using Base.Threads: @threads, nthreads

export read_logs_from_directory
export get_json_path

function get_json_path(data, value_path::Vector{String})::Union{Any, Nothing}
    current = data
    for key in value_path
        if !haskey(current, key)
            return nothing
        end
        current = current[key]
    end
    if typeof(current) == String
        try
            return parse(Int, current)
        catch
            try
                return parse(Float64, current)
            catch
                return current
            end
        end
    else
        return current
    end
end

function chunks(lst::Vector, n::Int)::Base.Generator
    # Yields successive n-sized chunks from lst.
    return (lst[i:min(i + n - 1, end)] for i in 1:n:length(lst))
end

function read_logs_from_directory(directory::String, value_paths::Dict{String, Vector{String}}, glob_pattern::String="*in*r*t*c*s*.json")::DataFrame
    log_path = joinpath(directory, "output")
    df_entries = Vector{Dict{String, Any}}()
    files = collect(glob(joinpath(log_path, glob_pattern)))

    @threads for file in files
        if filesize(file) == 0
            continue
        end
        log = open(file, "r") do io
            JSON.parse(read(io, String))
        end

        local_entries = Vector{Dict{String, Any}}()

        if isa(log, Vector)
            for (iteration, data) in enumerate(log)
                entry = Dict{String, Any}()
                for (name, path) in value_paths
                    value = get_json_path(data, path)
                    if isa(value, Vector) && length(value) == 1
                        value = value[1]
                    end
                    entry[name] = value
                end
                entry["iteration"] = iteration - 1
                push!(local_entries, entry)
            end
        else
            base_entry = Dict{String, Any}()
            to_expand = Dict{String, Vector}()
            for (name, path) in value_paths
                value = get_json_path(log, path)
                if !isa(value, Vector)
                    base_entry[name] = value
                else
                    to_expand[name] = value
                end
            end
            iterations = get_json_path(log, ["info", "iterations"])
            if iterations === nothing
                max_iterations = maximum(length.(values(to_expand)))
                min_iterations = minimum(length.(values(to_expand)))
                @assert max_iterations == min_iterations
                iterations = max_iterations
            else
                iterations = parse(Int, iterations)
            end
            for iteration in 0:(iterations - 1)
                entry = deepcopy(base_entry)
                entry["iteration"] = iteration
                for (key, value) in to_expand
                    window_size = div(length(value), iterations)
                    window = value[(iteration * window_size) + 1 : ((iteration + 1) * window_size)]
                    if length(window) == 1
                        entry[key] = window[1]
                    else
                        entry[key] = window
                    end
                end
                push!(local_entries, entry)
            end
        end
        append!(df_entries, local_entries)
    end
    return DataFrame(df_entries)
end

end # module LogProcessing
