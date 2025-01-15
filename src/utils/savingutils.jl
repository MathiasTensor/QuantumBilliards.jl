using DataFrames, CSV,  PrettyTables

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
    df=DataFrame(k=ks,tension=tens)
    CSV.write(filename,df)
end

"""
    save_numerical_ks!(ks::Vector{T}, filename::String) where {T<:Real}

Saves numerical values of `ks` to a CSV file

# Arguments
- `ks::Vector{T}`: A vector of numerical eigenvalues `k`.
- `filename::String`: The name of the CSV file to save the data.

# Notes
- The function creates a `DataFrame` with a single column `k` and writes it to the specified `filename`.
"""
function save_numerical_ks!(ks::Vector{T}, filename::String) where {T<:Real}
    df=DataFrame(k=ks)
    CSV.write(filename,df)
end

"""
    save_numerical_ks_and_tensions_with_tags(ks::Vector{T}, tens::Vector{T}, k_start::T, k_end::T, filename::String) where {T<:Real}

Saves numerical eigenvalues `ks`, their tensions `tens`, and the range tags of the interval bounds (`k_start`, `k_end`) for later filtering into a CSV file. 
The output CSV file will contain four columns: `k` (eigenvalues), `tension` (tension values), `k1` (start of range), and `k2` (end of range).

# Arguments
- `ks::Vector{T}`: A vector of numerical eigenvalues.
- `tens::Vector{T}`: A vector of corresponding tension values.
- `k_start::T`: The starting value of the eigenvalue range for the file.
- `k_end::T`: The ending value of the eigenvalue range for the file.
- `filename::String`: The output filename for saving the data.
"""
function save_numerical_ks_and_tensions_with_tags(ks::Vector{T},tens::Vector{T},k_start::T,k_end::T,filename::String) where {T<:Real}
    tagged_df=DataFrame(k=ks,tension=tens,k1=fill(k_start,length(ks)),k_end=fill(k_end,length(ks)))
    CSV.write(filename,tagged_df)
end

"""
    read_numerical_ks_and_tensions_with_tags(filename::String)

Reads numerical eigenvalues, tensions, and range tags from a tagged CSV file. Assumes the input file contains columns `k`, `tension`, `k1`, and `k2`.

# Arguments
- `filename::String`: The input filename to read the data from.

# Returns
- `ks<:Real`: A vector of numerical eigenvalues.
- `tensions<:Real`: A vector of corresponding tension values.
- `k_start<:Real`: The starting value of the eigenvalue range for the file.
- `k_end<:Real`: The ending value of the eigenvalue range for the file.
"""
function read_numerical_ks_and_tensions_with_tags(filename::String) 
    tagged_df=CSV.read(filename,DataFrame)
    ks=tagged_df.k
    tensions=tagged_df.tension
    k_start=tagged_df.k1[1]
    k_end=tagged_df.k2[1]
    return ks,tensions,k_start,k_end
end

"""
    merge_and_filter_tagged_files(csv_files::Vector{String}, output_file::String; target_k_start::T, target_k_end::T) where {T<:Real}

Merges and filters eigenvalues and tensions from multiple tagged CSV files into a single output file. Usable when doing multi-node computation since each node's threads will work on only a part of the global interval (local interval). The merged and filtered data is sorted by eigenvalue before saving. The end result is a .csv file that has only columns `k` and `tension`.

# Arguments
- `csv_files::Vector{String}`: A vector of file paths to tagged CSV files.
- `output_file::String`: The output filename for saving the merged and filtered data.
- `target_k_start::T`: The global lower bound for eigenvalues to include, that is if we partition the global interval into local ones for redistributing.
- `target_k_end::T`: The global upper bound for eigenvalues to include.

# Returns
- `nothing`: The function does not return a value; instead, it saves the merged and filtered data to a NON-TAGGED (regular) csv file.
"""
function merge_and_filter_tagged_files!(csv_files::Vector{String},output_file::String;target_k_start::T,target_k_end::T) where {T<:Real}
    all_ks=T[] # unknown size
    all_tensions=T[]
    for csv_file in csv_files
        try
            df=CSV.read(csv_file,DataFrame)
            if !(haskey(df,:k1) && haskey(df,:k2))
                println("$csv_file is not a tagged file, skipping")
                continue
            end
            ks,tensions,k_start,k_end=read_numerical_ks_and_tensions_with_tags(csv_file)
            # make a global check and local check for k bounds (global for whole set of files and local for the file in process)
            valid_indices_global=findall(k -> k>=target_k_start && k<=target_k_end,ks) 
            if isempty(valid_indices_global)
                println("Skipping file: $(basename(csv_file)) (all ks out of global target interval), problem!")
                continue
            end
            ks=ks[valid_indices_global]
            tensions=tensions[valid_indices_global]
            valid_indices_local=findall(k -> k>=k_start && k<=k_end,ks) 
            if isempty(valid_indices_local)
                println("Skipping file: $(basename(csv_file)) (Relevant file empty in local interval)")
                continue
            end
            ks=ks[valid_indices_local]
            tensions=tensions[valid_indices_local]
            append!(all_ks,ks)
            append!(all_tensions,tensions)
        catch e
            println("Error processing file $(basename(csv_file)): ", e)
        end
    end
    # Sort b/c can be arbitrary ordering
    sorted_indices=sortperm(all_ks)
    all_ks=all_ks[sorted_indices]
    all_tensions=all_tensions[sorted_indices]
    save_numerical_ks_and_tensions!(all_ks,all_tensions,output_file)
    println("Merged and filtered data saved to: $output_file")
end

"""
    merge_and_filter_tagged_files(directory::String, output_file::String; target_k_start::T, target_k_end::T) where {T<:Real}

Merges and filters eigenvalues and tensions from all tagged CSV files in a directory into a single output file.

# Arguments
- `directory::String`: Path to the directory containing tagged CSV files.
- `output_file::String`: The output filename for saving the merged and filtered data.
- `target_k_start::T`: The global lower bound for eigenvalues to include.
- `target_k_end::T`: The global upper bound for eigenvalues to include.

# Returns
- Nothing
"""
function merge_and_filter_tagged_files!(directory::String,output_file::String;target_k_start::T,target_k_end::T) where {T<:Real}
    # Collect all `.csv` files in the directory
    csv_files=glob("*.csv",directory)
    if isempty(csv_files)
        println("No CSV files found in directory: $directory")
        return
    end
    println("Processing $(length(csv_files)) files from directory: $directory")
    merge_and_filter_from_files(csv_files, output_file; target_k_start=target_k_start, target_k_end=target_k_end)
end

"""
    save_numerical_ks_and_overlaps(ks::Vector{T}, Ms::Vector{T}, filename::String) where {T<:Real}

Saves numerical eigenvalues and their corresponding overlaps to a CSV file.

# Arguments
- `ks::Vector{T}`: A vector of numerical eigenvalues `k`.
- `Ms::Vector{T}`: A vector of overlap values corresponding to each eigenvalue.
- `filename::String`: The name of the CSV file to save the data.

# Notes
- The function creates a `DataFrame` with two columns: 
  - `k`: The eigenvalues.
  - `M`: The overlaps.
- Writes the `DataFrame` to the specified `filename` in CSV format.
"""
function save_numerical_ks_and_overlaps(ks::Vector{T},Ms::Vector{T},filename::String) where {T<:Real}
    df=DataFrame(k=ks,M=Ms)
    CSV.write(filename,df)
end

"""
    read_numerical_ks_and_overlaps(filename::String) -> (ks, Ms)

Reads numerical eigenvalues and their corresponding tensions from a CSV file.

# Arguments
- `filename::String`: The name of the CSV file to read the data from.

# Returns
- `ks`: A vector of numerical eigenvalues read from the `k` column of the file.
- `tensions`: A vector of overlaps read from the `M` column of the file.

# Notes
- CSV file has columns named `k` (for eigenvalues) and `M` (for overlaps).
- The function returns the data as two separate vectors, `ks` and `Ms`.
"""
function read_numerical_ks_and_overlaps(filename::String)
    df=CSV.read(filename,DataFrame)
    ks=df.k
    Ms=df.M
    return ks,Ms
end

"""
    save_numerical_ks_and_localizations(ks::Vector{T}, entropies::Vector{T}, filename::String) where {T<:Real}

Saves numerical eigenvalues and their corresponding localization entropies to a CSV file.

# Arguments
- `ks::Vector{T}`: A vector of numerical eigenvalues `k`.
- `Ms::Vector{T}`: A vector of localization entropy values corresponding to each eigenvalue.
- `filename::String`: The name of the CSV file to save the data.

# Notes
- The function creates a `DataFrame` with two columns: 
  - `k`: The eigenvalues.
  - `A`: The entropies.
- Writes the `DataFrame` to the specified `filename` in CSV format.
"""
function save_numerical_ks_and_localizations(ks::Vector{T}, entropies::Vector{T}, filename::String) where {T<:Real}
    df=DataFrame(k=ks,A=entropies)
    CSV.write(filename,df)
end

"""
    read_numerical_ks_and_localizations(filename::String) -> (ks, Ms)

Reads numerical eigenvalues and their corresponding tensions from a CSV file.

# Arguments
- `filename::String`: The name of the CSV file to read the data from.

# Returns
- `ks`: A vector of numerical eigenvalues read from the `k` column of the file.
- `tensions`: A vector of entropies read from the `A` column of the file.

# Notes
- CSV file has columns named `k` (for eigenvalues) and `A` (for entropies).
- The function returns the data as two separate vectors, `ks` and `As`.
"""
function read_numerical_ks_and_localizations(filename::String)
    df=CSV.read(filename,DataFrame)
    ks=df.k
    As=df.A
    return ks,As
end

"""
    read_numerical_ks_and_tensions(filename::String) -> (ks, tensions)

Read numerical eigenvalues and tensions from a CSV file.

# Arguments
- `filename::String`: Name of the CSV file to read the data from.

# Returns
- `ks`: Vector of numerical eigenvalues. This is a SentinelArray so you should use Vector(...) on the ks and tensions.
- `tensions`: Vector of tensions corresponding to each eigenvalue.

# Notes
- Assumes the CSV file has columns named `k` and `tension`.
"""
function read_numerical_ks_and_tensions(filename::String)
    df=CSV.read(filename,DataFrame)
    ks=df.k
    tensions=df.tension
    return ks,tensions
end

"""
    read_numerical_ks(filename::String)

Read numerical eigenvalues from a CSV file.

# Arguments
- `filename::String`: Name of the CSV file to read the data from.

# Returns
- `ks`: Vector of numerical eigenvalues. This is a SentinelArray so you should use Vector(...) on the ks and tensions.

# Notes
- Assumes the CSV file has a single column named `k`.
"""
function read_numerical_ks(filename::String)
    df=CSV.read(filename,DataFrame)
    ks=df.k
    return ks
end

"""
    filter_and_save_ks_and_tensions(input_filename::String, output_filename::String, k_min::T, k_max::T) where {T<:Real}

Filters a read .csv file containing the `k` and `tensions` header using the `k_min` and `k_max` as the lower and upper bounds and then saves it.

# Arguments
- `input_filename::String`: Name of the input CSV file containing the `k` and `tension` header.
- `output_filename::String`: Name of the output CSV file to save the filtered data.
- `k_min::T`: Lower bound for the `k` values.
- `k_max::T`: Upper bound for the `k` values.

# Returns
`Nothing`
"""
function filter_and_save_ks_and_tensions!(input_filename::String,output_filename::String,k_min::T,k_max::T) where {T<:Real}
    df=CSV.read(input_filename,DataFrame)
    filtered_df=filter(row->k_min<=row.k<=k_max,df)
    CSV.write(output_filename,filtered_df)
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
function compute_and_save_closest_pairs!(ksA::Vector{T},ksB::Vector{T},A::T,filename::String;unique=true,tolerance::Float64=1e-6) where T<:Real
    function filter_unique_by_tolerance(v::Vector{T})
        sorted_v=sort(v)
        filtered=[sorted_v[1]]
        for i in eachindex(sorted_v)[2:end]
            if abs(sorted_v[i]-filtered[end])>tolerance
                push!(filtered,sorted_v[i])
            end
        end
        return filtered
    end
    if unique
        ksA=filter_unique_by_tolerance(ksA)
    end
    results=[]
    for kA in ksA
        # Find the closest element in ksB
        kB=ksB[argmin(abs.(ksB.-kA))]  # Find kB with the smallest absolute difference
        # Compute absolute and relative differences
        abs_diff=abs(kA-kB)
        relative_diff=abs_diff/abs(kA)
        diff_A_term=(kA^2-kB^2)/(1/(A/(4*π)))
        is_below_threshold=abs(diff_A_term)<0.01
        push!(results,(kA,kB,relative_diff,abs_diff,diff_A_term,is_below_threshold))
    end
    df=DataFrame(kA=[r[1] for r in results],kB=[r[2] for r in results],relative_diff=[r[3] for r in results],abs_diff=[r[4] for r in results],diff_A_term=[r[5] for r in results],below_threshold=[r[6] for r in results])
    CSV.write(filename*".csv",df)
    println("Closest pairs and analysis written to $filename")
    header=["k_num","k_anal","Relative Diff","Absolute Diff","< 0.01","Below Thresh (1/0)?"]
    data_matrix=hcat(df.kA,df.kB,df.relative_diff,df.abs_diff,df.diff_A_term,df.below_threshold)
    # Save as a LaTeX file using PrettyTables
    open(filename*".tex","w") do f
        pretty_table(f,data_matrix,header=header,backend=Val(:latex))
    end
end
