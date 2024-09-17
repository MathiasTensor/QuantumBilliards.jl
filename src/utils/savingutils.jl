using DataFrames, CSV,  PrettyTables


#=
DrWatson._wsave(filename, df::DataFrame; kwargs...) = Arrow.write(filename, df; kwargs...)

function save_kspectrum(spect_data, params; folder="spectra")
    output_stream = datadir(folder, savename("kspectrum", params, "arrow"; sigdigits = 4))
    #println(output_stream)
    df_spect =  DataFrame(k=spect_data.k, ten=spect_data.ten, control=spect_data.control)
    safesave(output_stream, df_spect)
end


function load_kspectrum(params; folder="spectra")
    output_stream = datadir(folder, savename("kspectrum", params, "arrow"; sigdigits = 4))
    #println(output_stream)
    df_spect =  DataFrame(k=spect_data.k, ten=spect_data.ten, control=spect_data.control)
    safesave(output_stream, df_spect)
end
=#




######## NEW ADDITIONS #########





"""
    save_numerical_ks_and_tensions!(ks::Vector{T}, tens::Vector{T}, filename::String) where {T<:Real}

Saves numerical values of `ks` (eigenvalues) and `tensions` to a CSV file.

# Arguments
- `ks::Vector{T}`: A vector of numerical eigenvalues `k`.
- `tens::Vector{T}`: A vector of corresponding tension values.
- `filename::String`: The name of the CSV file to save the data.

# Notes
- The function creates a `DataFrame` with columns `k` and `tension` and writes it to the specified `filename`.
"""
function save_numerical_ks_and_tensions!(ks::Vector{T}, tens::Vector{T}, filename::String) where {T<:Real}
    df = DataFrame(k=ks, tension=tens)
    CSV.write(filename, df)
end



"""
    read_numerical_ks_and_tensions(filename::String) -> (ks, tensions)

Read numerical eigenvalues and tensions from a CSV file.

# Arguments
- `filename::String`: Name of the CSV file to read the data from.

# Returns
- `ks`: Vector of numerical eigenvalues.
- `tensions`: Vector of tensions corresponding to each eigenvalue.

# Notes
- Assumes the CSV file has columns named `k` and `tension`.
"""
function read_numerical_ks_and_tensions(filename::String)
    df = CSV.read(filename, DataFrame)
    ks = df.k
    tensions = df.tension
    return ks, tensions
end



"""
    compute_and_save_closest_pairs!(ksA::Vector{T}, ksB::Vector{T}, A::T, filename::String; unique=true, tolerance::Float64=1e-6) where {T<:Real}

Compute and save the closest pairs between numerical and analytical eigenvalues.

# Arguments
- `ksA::Vector{T}`: Vector of numerical eigenvalues.
- `ksB::Vector{T}`: Vector of analytical eigenvalues.
- `A::T`: Constant value used in the calculation of `diff_A_term`.
- `filename::String`: Base name for the output files (CSV and LaTeX).

# Keyword Arguments
- `unique::Bool=true`: If `true`, filters out duplicate eigenvalues within a specified `tolerance`.
- `tolerance::Float64=1e-6`: Tolerance used when filtering for unique eigenvalues.

# Description
- For each numerical eigenvalue in `ksA`, finds the closest analytical eigenvalue in `ksB`.
- Computes the absolute difference, relative difference, and a custom difference term `diff_A_term`.
- Determines if `diff_A_term` is below a threshold of `0.01`.
- Saves the results to a CSV file and generates a LaTeX table for documentation.

# Outputs
- CSV file: `filename.csv` containing the computed data.
- LaTeX file: `filename.tex` containing the data formatted as a LaTeX table.

# Notes
- The function is intended to match numerical eigenvalues (`ksA`) with analytical eigenvalues (`ksB`).
- The `diff_A_term` is calculated as `(kA^2 - kB^2) / (1 / (A / (4 * π)))`.
"""
function compute_and_save_closest_pairs!(ksA::Vector{T}, ksB::Vector{T}, A::T, filename::String; unique=true, tolerance::Float64 = 1e-6) where T<:Real
    function filter_unique_by_tolerance(v::Vector{T})
        sorted_v = sort(v)
        filtered = [sorted_v[1]]
        for i in eachindex(sorted_v)[2:end]
            if abs(sorted_v[i] - filtered[end]) > tolerance
                push!(filtered, sorted_v[i])
            end
        end
        return filtered
    end

    if unique
        ksA = filter_unique_by_tolerance(ksA)
    end

    results = []
    for kA in ksA
        # Find the closest element in ksB
        kB = ksB[argmin(abs.(ksB .- kA))]  # Find kB with the smallest absolute difference
        # Compute absolute and relative differences
        abs_diff = abs(kA - kB)
        relative_diff = abs_diff / abs(kA)
        diff_A_term = (kA^2 - kB^2) / (1/(A / (4 * π)))
        is_below_threshold = abs(diff_A_term) < 0.01
        push!(results, (kA, kB, relative_diff, abs_diff, diff_A_term, is_below_threshold))
    end
    df = DataFrame(kA=[r[1] for r in results], kB=[r[2] for r in results],
                   relative_diff=[r[3] for r in results], abs_diff=[r[4] for r in results],
                   diff_A_term=[r[5] for r in results], below_threshold=[r[6] for r in results])
    CSV.write(filename * ".csv", df)
    println("Closest pairs and analysis written to $filename")

    header = ["k_num", "k_anal", "Relative Diff", "Absolute Diff", "< 0.01", "Below Thresh (1/0)?"]
    data_matrix = hcat(df.kA, df.kB, df.relative_diff, df.abs_diff, df.diff_A_term, df.below_threshold)
    
    # Save as a LaTeX file using PrettyTables
    open(filename * ".tex", "w") do f
        pretty_table(f, data_matrix, header=header, backend=Val(:latex))
    end
end







