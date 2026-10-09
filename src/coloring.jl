##
# Column coloring of sparsity patterns for the assembly of sparse Jacobians
##

# Distance-2 greedy coloring of the columns of `P` in the given `order`: each column gets
# the smallest color not used by a column with a nonzero in a common row. `Pt` is the
# transpose of `P` (the nonzero columns of each row). `forbidden[c] == j` marks color `c` as
# used by a neighbor of column `j`.
function greedy_coloring!(colors, forbidden, P::SparseMatrixCSC, Pt::SparseMatrixCSC, order)
    rows = rowvals(P)
    cols_of_row = rowvals(Pt)
    fill!(colors, 0)
    fill!(forbidden, 0)
    for j in order
        for idx in nzrange(P, j)
            for idx2 in nzrange(Pt, rows[idx])
                c = colors[cols_of_row[idx2]]
                c > 0 && (forbidden[c] = j)
            end
        end
        c = 1
        while forbidden[c] == j
            c += 1
        end
        colors[j] = c
    end
    return colors
end

# Vertex orders for the greedy coloring
function coloring_order(order::Symbol, P::SparseMatrixCSC, Pt::SparseMatrixCSC)
    n = size(P, 2)
    if order === :natural
        return 1:n
    elseif order === :reverse
        return n:-1:1
    elseif order === :largest_first
        # Upper bound of the number of neighbors: the sum of the nonzeros of the rows of
        # the column (cheap compared with the exact distance-2 degree)
        row_nnz = [length(nzrange(Pt, i)) for i in 1:size(P, 1)]
        rows = rowvals(P)
        degree = [sum(i -> row_nnz[rows[i]], nzrange(P, j); init = 0) for j in 1:n]
        return sortperm(degree; rev = true, alg = Base.Sort.DEFAULT_STABLE)
    else
        throw(ArgumentError("unknown coloring order $(repr(order)), use :natural, :reverse, :largest_first, or a permutation vector"))
    end
end
coloring_order(order::AbstractVector{<:Integer}, P, Pt) = order

function bool_pattern(pattern::AbstractMatrix)
    P = SparseMatrixCSC{Bool, Int}(sparse(pattern) .!= 0)
    dropzeros!(P)
    return P
end

"""
    greedy_column_coloring(pattern; order = :natural) -> colors::Vector{Int}

Greedy distance-2 coloring of the columns of the sparsity `pattern`: two columns get
different colors if they have a nonzero in a common row. Columns of the same color are
structurally orthogonal and can be computed by a single Jacobian-vector product.

The columns are colored in the `order` `:natural`, `:reverse`, `:largest_first` (by an
upper bound of the number of neighbors), or the given permutation vector.

Other algorithms are available in SparseMatrixColorings.jl, e.g.,
`column_colors(coloring(pattern, ColoringProblem(), GreedyColoringAlgorithm(SmallestLast())))`.
"""
function greedy_column_coloring(pattern::AbstractMatrix; order = :natural)
    P = bool_pattern(pattern)
    Pt = sparse(transpose(P))
    n = size(P, 2)
    return greedy_coloring!(zeros(Int, n), zeros(Int, n + 1), P, Pt, coloring_order(order, P, Pt))
end

# Columns grouped by color: `cols[ptr[c]:(ptr[c + 1] - 1)]` are the columns of color `c`
function color_groups(colors::AbstractVector{<:Integer}, ncolors = maximum(colors; init = 0))
    ptr = zeros(Int, ncolors + 1)
    for c in colors
        ptr[c + 1] += 1
    end
    ptr[1] = 1
    cumsum!(ptr, ptr)
    cols = Vector{Int}(undef, length(colors))
    next = ptr[1:ncolors]
    for (j, c) in enumerate(colors)
        cols[next[c]] = j
        next[c] += 1
    end
    return ptr, cols
end

"""
    is_column_coloring(pattern, colors) -> Bool

Whether the columns of each color of `colors` are structurally orthogonal in `pattern`,
i.e., no two columns of the same color have a nonzero in a common row.
"""
function is_column_coloring(pattern::AbstractMatrix, colors::AbstractVector{<:Integer})
    P = bool_pattern(pattern)
    length(colors) == size(P, 2) || return false
    all(>(0), colors) || return false
    Pt = sparse(transpose(P))
    seen = zeros(Int, maximum(colors; init = 0))
    cols_of_row = rowvals(Pt)
    for i in 1:size(P, 1)
        for idx in nzrange(Pt, i)
            c = colors[cols_of_row[idx]]
            seen[c] == i && return false
            seen[c] = i
        end
    end
    return true
end

"""
    column_coloring_lower_bound(pattern) -> Int

Lower bound of the number of colors of any column coloring of `pattern`: the maximum
number of nonzeros in a row (all columns of a row need different colors).
"""
function column_coloring_lower_bound(pattern::AbstractMatrix)
    Pt = sparse(transpose(bool_pattern(pattern)))
    return maximum(j -> length(nzrange(Pt, j)), 1:size(Pt, 2); init = 0)
end
