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
upper bound of the number of neighbors), or the given permutation vector. See
[`column_coloring`](@ref) for colorings that try several orders and improve the best
coloring by iterated greedy recoloring.

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

# Culberson's iterated greedy: recolor the columns greedily, ordered by color classes.
# Since the columns of each class are independent, the greedy coloring in such an order
# never needs more colors. The classes are ordered by reverse color, by decreasing size,
# or randomly, in turn.
function recolor!(colors, P, Pt, iterations, rng)
    n = size(P, 2)
    forbidden = zeros(Int, n + 1)
    best = copy(colors)
    nbest = maximum(colors; init = 0)
    current = copy(colors)
    for iteration in 1:iterations
        k = maximum(current; init = 0)
        ptr, cols = color_groups(current, k)
        class_order = if iteration % 3 == 1
            k:-1:1
        elseif iteration % 3 == 2
            sortperm(diff(ptr); rev = true)
        else
            randperm(rng, k)
        end
        order = reduce(vcat, (view(cols, ptr[c]:(ptr[c + 1] - 1)) for c in class_order); init = Int[])
        greedy_coloring!(current, forbidden, P, Pt, order)
        k = maximum(current; init = 0)
        if k < nbest
            nbest = k
            copyto!(best, current)
        end
    end
    copyto!(colors, best)
    return colors
end

"""
    column_coloring(pattern; orders = (:natural, :reverse, :largest_first),
                    recolor_iterations = 0, rng = Random.Xoshiro(0)) -> colors::Vector{Int}

Distance-2 column coloring of the sparsity `pattern` for the assembly of sparse Jacobians
(see [`greedy_column_coloring`](@ref)). The greedy coloring is computed for each of the
`orders`, the one with the fewest colors is kept and then improved by
`recolor_iterations` passes of Culberson's iterated greedy algorithm (which never increases
the number of colors). Each pass costs about as much as one greedy coloring.

The gains depend on the pattern. For the Jacobians of discontinuous Galerkin
discretizations on structured meshes, the natural order can already be within a few
percent of the best of these orders and many recoloring passes; for such patterns, see
[`quotient_coloring`](@ref), which exploits the block structure.
"""
function column_coloring(
        pattern::AbstractMatrix; orders = (:natural, :reverse, :largest_first), recolor_iterations::Integer = 0,
        rng = Random.Xoshiro(0)
    )
    P = bool_pattern(pattern)
    Pt = sparse(transpose(P))
    n = size(P, 2)
    forbidden = zeros(Int, n + 1)
    colors = zeros(Int, n)
    best = Int[]
    for order in orders
        greedy_coloring!(colors, forbidden, P, Pt, coloring_order(order, P, Pt))
        if isempty(best) || maximum(colors; init = 0) < maximum(best; init = 0)
            best = copy(colors)
        end
    end
    isempty(best) && (best = greedy_coloring!(colors, forbidden, P, Pt, 1:n))
    return recolor!(best, P, Pt, recolor_iterations, rng)
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

# Greedy coloring of the vertices of the graph with the symmetric adjacency pattern `G`
# (the diagonal is ignored) in the given `order`
function greedy_graph_coloring!(colors, forbidden, G::SparseMatrixCSC, order)
    neighbors = rowvals(G)
    fill!(colors, 0)
    fill!(forbidden, 0)
    for v in order
        for idx in nzrange(G, v)
            c = colors[neighbors[idx]]
            c > 0 && (forbidden[c] = v)
        end
        c = 1
        while forbidden[c] == v
            c += 1
        end
        colors[v] = c
    end
    return colors
end

# Best greedy coloring of the graph `G` over several orders, each improved by iterated greedy
function multistart_graph_coloring(G::SparseMatrixCSC, trials, recolor_iterations, rng)
    nq = size(G, 2)
    degree = [length(nzrange(G, v)) for v in 1:nq]
    forbidden = zeros(Int, nq + 1)
    colors = zeros(Int, nq)
    best = Int[]
    for trial in 1:max(trials, 1)
        order = if trial == 1
            1:nq
        elseif trial == 2
            sortperm(degree; rev = true)
        else
            randperm(rng, nq)
        end
        greedy_graph_coloring!(colors, forbidden, G, order)
        for iteration in 1:recolor_iterations
            k = maximum(colors; init = 0)
            ptr, members = color_groups(colors, k)
            class_order = isodd(iteration) ? (k:-1:1) : randperm(rng, k)
            order = reduce(vcat, (view(members, ptr[c]:(ptr[c + 1] - 1)) for c in class_order); init = Int[])
            greedy_graph_coloring!(colors, forbidden, G, order)
        end
        if isempty(best) || maximum(colors; init = 0) < maximum(best; init = 0)
            best = copy(colors)
        end
    end
    return best
end

"""
    quotient_coloring(pattern, classes; trials = 20, recolor_iterations = 30,
                      rng = Random.Xoshiro(0)) -> colors::Vector{Int}

Column coloring of `pattern` in which all columns `j` with the same label `classes[j]` get
the same color. Two classes conflict if any of their columns have a nonzero in a common
row; the (small) conflict graph of the classes is colored greedily in `trials` orders
(natural, largest degree first, and random), each improved by `recolor_iterations` passes
of iterated greedy, and the best coloring is kept. Throws an `ArgumentError` if two
columns of the same class have a nonzero in a common row.

This exploits block and translation structure that a greedy coloring of the columns does
not see. For discontinuous Galerkin discretizations on structured meshes, label column
`j` by its local index in the element and the position of the element modulo a small
periodic cell of elements, see [`block_classes`](@ref).
"""
function quotient_coloring(
        pattern::AbstractMatrix, classes::AbstractVector{<:Integer}; trials::Integer = 20,
        recolor_iterations::Integer = 30, rng = Random.Xoshiro(0)
    )
    P = bool_pattern(pattern)
    m, n = size(P)
    length(classes) == n || throw(ArgumentError("classes must have one entry per column"))
    labels = unique(classes)
    index = Dict(c => k for (k, c) in enumerate(labels))
    q = [index[c] for c in classes]
    nq = length(labels)
    # Q[i, k]: row `i` has a nonzero in a column of class `k`
    S = sparse(1:n, q, true, n, nq)
    Q = P * S
    nnz(Q) == nnz(P) || throw(ArgumentError("two columns of the same class have a nonzero in a common row"))
    Qb = SparseMatrixCSC{Bool, Int}(Q .!= 0)
    G = SparseMatrixCSC{Bool, Int}(transpose(Qb) * Qb .!= 0)
    colors_q = multistart_graph_coloring(G, trials, recolor_iterations, rng)
    return colors_q[q]
end

"""
    block_classes(block_size, block_labels) -> classes::Vector{Int}

Class labels for [`quotient_coloring`](@ref) of the columns of a block-structured pattern
with consecutive blocks of `block_size` columns (e.g., the degrees of freedom of the
elements of a discontinuous Galerkin discretization): column `j` in block `e` at local
index `ℓ` gets the label of the pair `(ℓ, block_labels[e])`. For example, for elements on a
Cartesian grid, `block_labels[e] = 1 + mod(i, a) + a * mod(j, b)` with the element
indices `(i, j)` uses a periodic cell of `a × b` elements.
"""
function block_classes(block_size::Integer, block_labels::AbstractVector{<:Integer})
    index = Dict(c => k for (k, c) in enumerate(unique(block_labels)))
    return [ℓ + block_size * (index[c] - 1) for c in block_labels for ℓ in 1:block_size]
end
