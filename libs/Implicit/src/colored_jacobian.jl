using SparseDiffTools
using LinearAlgebra
using SparseArrays

##################### BatchedJacobianOperator

struct BatchedColoredUpdater{T}
    colors::Vector{Int}
    color_groups::Vector{Vector{Int}}
    sparse_template::SparseMatrixCSC{T, Int}
    input_matrix::Matrix{T}        # Pre-allocata
    result_matrix::Matrix{T}       # Pre-allocata
    extraction_plan::Vector{Vector{Tuple{Int, Int}}}
end

function BatchedColoredUpdater(Jsp::SparseMatrixCSC{T}) where T
    pattern = sparse((Jsp .!= 0) * 1.0)
    colors = matrix_colors(pattern)
    
    color_groups = [Int[] for _ in 1:maximum(colors)]
    for (col, color) in enumerate(colors)
        push!(color_groups[color], col)
    end
    
    n = size(Jsp, 2)
    m = size(Jsp, 1)
    n_groups = length(color_groups)
    
    # Pre-alloca le matrici
    input_matrix = zeros(T, n, n_groups)
    result_matrix = zeros(T, m, n_groups)
    
    # Setup input matrix una volta sola
    for (group_idx, cols) in enumerate(color_groups)
        for col in cols
            input_matrix[col, group_idx] = 1.0
        end
    end
    
    # Pre-compute extraction plan
    extraction_plan = Vector{Vector{Tuple{Int, Int}}}(undef, n_groups)
    for (group_idx, cols) in enumerate(color_groups)
        plan = Tuple{Int, Int}[]
        for col in cols
            for i in Jsp.colptr[col]:(Jsp.colptr[col+1]-1)
                row = Jsp.rowval[i]
                push!(plan, (i, row))
            end
        end
        extraction_plan[group_idx] = plan
    end
    
    return BatchedColoredUpdater(colors, color_groups, copy(Jsp),
                                input_matrix, result_matrix, extraction_plan)
end

 function update_sparse_batched!(updater::BatchedColoredUpdater, J_batched)
           # UNA SOLA moltiplicazione batched per tutti i 67 gruppi!
           mul!(updater.result_matrix, J_batched, updater.input_matrix)

           # Estrazione veloce usando il plan pre-computato
           nzval = updater.sparse_template.nzval

           @inbounds for (group_idx, plan) in enumerate(updater.extraction_plan)
               result_col = view(updater.result_matrix, :, group_idx)

               # Rimuovi @simd quando iteriamo su tuple
               for (nz_idx, res_idx) in plan
                   nzval[nz_idx] = result_col[res_idx]
               end
           end

    return updater.sparse_template
end
