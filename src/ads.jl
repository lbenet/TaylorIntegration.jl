# This file is part of the TaylorIntegration.jl package; MIT licensed

## Constructors

"""
    ADSTaylorSolution{T, N, M} <: AbstractTaylorSolution{T, TaylorN{T}}

This `struct` is modified in-place in the ADS method of `taylorinteg`, and represents
a node in a binary tree. Fields `t` and `x` represent, respectively, the value of time
(independent variable) and the computed values of the dependent variables(s) at the
current node. When `taylorinteg` is called with `dense=true`, then field `p` stores the
Taylor polynomial expansion at the current node. Fields `lo` and `hi` correspond,
respectively, to the lower and upper bounds of the jet transport domain. Fields `parent`,
`left` and `right` are references to the current node's connections in the binary tree.
"""
mutable struct ADSTaylorSolution{T, N, M} <: AbstractTaylorSolution{T, TaylorN{T}}
    depth::Int
    t::T
    lo::SVector{N, T}
    hi::SVector{N, T}
    x::SVector{M, TaylorN{T}}
    p::Union{Nothing, SVector{M, Taylor1{TaylorN{T}}}}
    parent::Union{Nothing, ADSTaylorSolution{T, N, M}}
    left::Union{Nothing, ADSTaylorSolution{T, N, M}}
    right::Union{Nothing, ADSTaylorSolution{T, N, M}}
    function ADSTaylorSolution{T, N, M}(depth::Int, t::T, lo::SVector{N, T},
        hi::SVector{N, T}, x::SVector{M, TaylorN{T}},
        p::Union{Nothing, SVector{M, Taylor1{TaylorN{T}}}},
        parent::Union{Nothing, ADSTaylorSolution{T, N, M}},
        left::Union{Nothing, ADSTaylorSolution{T, N, M}} = nothing,
        right::Union{Nothing, ADSTaylorSolution{T, N, M}} = nothing) where {T, N, M}
        @assert all(hi .> lo)
        new{T, N, M}(depth, t, lo, hi, x, p, parent, left, right)
    end
end

ADSTaylorSolution(depth::Int, t::T, lo::SVector{N, T},
    hi::SVector{N, T}, x::SVector{M, TaylorN{T}},
    p::Union{Nothing, SVector{M, Taylor1{TaylorN{T}}}},
    parent::Union{Nothing, ADSTaylorSolution{T, N, M}},
    left::Union{Nothing, ADSTaylorSolution{T, N, M}},
    right::Union{Nothing, ADSTaylorSolution{T, N, M}}) where {T, N, M} =
    ADSTaylorSolution{T, N, M}(depth, t, lo, hi, x, p, parent, left, right)

# 3-arg constructor
function ADSTaylorSolution(lo::AbstractVector{T}, hi::AbstractVector{T},
    x::AbstractVector{TaylorN{T}}) where {T}
    @assert length(lo) == length(hi)
    N = length(lo)
    M = length(x)
    return ADSTaylorSolution(0, zero(T), SVector{N, T}(lo), SVector{N, T}(hi),
    SVector{M, TaylorN{T}}(x), nothing, nothing, nothing, nothing)
end

## Custom print

function Base.show(io::IO, n::ADSTaylorSolution{T, N, M}) where {T, N, M}
    s = Vector{String}(undef, N)
    for i in eachindex(s)
        s[i] = string(SVector{2, T}(n.lo[i], n.hi[i]))
    end
    plural = M > 1 ? "s" : ""
    print(io, "t: ", n.t, " s: ", join(s, "×"), " x: ", M, " ",
        TaylorN{T}, " variable" * plural)
end

## AbstractTrees API

# Left (right) child constructors
function leftchild!(
    parent::ADSTaylorSolution{T, N, M}, t::T, lo::SVector{N, T},
    hi::SVector{N, T}, x::SVector{M, TaylorN{T}},
    p::Union{Nothing, SVector{M, Taylor1{TaylorN{T}}}} = nothing,
    left::Union{Nothing, ADSTaylorSolution{T, N, M}} = nothing,
    right::Union{Nothing, ADSTaylorSolution{T, N, M}} = nothing) where {T, N, M}
    isnothing(parent.left) || error("Left child is already assigned")
    node = ADSTaylorSolution{T, N, M}(parent.depth + 1, t, lo, hi, x, p, parent,
                                      left, right)
    parent.left = node
end

function leftchild!(parent::ADSTaylorSolution{T, N, M},
    node::ADSTaylorSolution{T, N, M}) where {T, N, M}
    # set `node` as left child of `parent`
    parent.left = node
end

function rightchild!(
    parent::ADSTaylorSolution{T, N, M}, t::T, lo::SVector{N, T},
    hi::SVector{N, T}, x::SVector{M, TaylorN{T}},
    p::Union{Nothing, SVector{M, Taylor1{TaylorN{T}}}} = nothing,
    left::Union{Nothing, ADSTaylorSolution{T, N, M}} = nothing,
    right::Union{Nothing, ADSTaylorSolution{T, N, M}} = nothing) where {T, N, M}
    isnothing(parent.right) || error("Right child is already assigned")
    node = ADSTaylorSolution{T, N, M}(parent.depth + 1, t, lo, hi, x, p, parent,
                                      left, right)
    parent.right = node
end

function rightchild!(parent::ADSTaylorSolution{T, N, M},
    node::ADSTaylorSolution{T, N, M}) where {T, N, M}
    # set `node` as right child of `parent`
    parent.right = node
end

# AbstractTrees interface
function AbstractTrees.children(node::ADSTaylorSolution)
    if isnothing(node.left) && isnothing(node.right)
        ()
    elseif isnothing(node.left) && !isnothing(node.right)
        (node.right,)
    elseif !isnothing(node.left) && isnothing(node.right)
        (node.left,)
    else
        (node.left, node.right)
    end
end

function AbstractTrees.printnode(io::IO, n::ADSTaylorSolution{T, N, M}) where {T, N, M}
    s = Vector{String}(undef, N)
    for i in eachindex(s)
        s[i] = string(SVector{2, T}(n.lo[i], n.hi[i]))
    end
    plural = M > 1 ? "s" : ""
    print(io, "t: ", n.t, " s: ", join(s, "×"), " x: ", M, " ",
        TaylorN{T}, " variable" * plural)
end

AbstractTrees.nodevalue(n::ADSTaylorSolution) = (n.t, n.lo, n.hi, n.x, n.p)

AbstractTrees.ParentLinks(::Type{<:ADSTaylorSolution}) = StoredParents()

AbstractTrees.parent(n::ADSTaylorSolution) = n.parent

AbstractTrees.NodeType(::Type{<:ADSTaylorSolution}) = HasNodeType()
AbstractTrees.nodetype(::Type{<:ADSTaylorSolution{T, N, M}}) where {T, N, M} =
    ADSTaylorSolution{T, N, M}

# AbstractTrees iteration interface
# The overload of Base.IteratorEltype may be redundant but is kept following the note in:
# https://juliacollections.github.io/AbstractTrees.jl/stable/iteration/#Interface
Base.IteratorEltype(::Type{<:TreeIterator{ADSTaylorSolution}}) = HasEltype()
Base.eltype(::Type{<:TreeIterator{ADSTaylorSolution{T, N, M}}}) where {T, N, M} =
    ADSTaylorSolution{T, N, M}

## Evaluation

"""
    countnodes(n::ADSTaylorSolution{T, N, M}, k::Int) where {T <: Real, N, M}
    countnodes(n::ADSTaylorSolution{T, N, M}, t::T) where {T <: Real, N, M}

Return the number of nodes at depth `k` (time `t`) starting from root node `n`.
"""
function countnodes(n::ADSTaylorSolution{T, N, M}, k::Int) where {T <: Real, N, M}
    s = 0
    for node in PreOrderDFS(x -> x.depth <= k, n)
        s += (node.depth == k)
    end
    return s
end

function countnodes(n::ADSTaylorSolution{T, N, M}, t::T) where {T <: Real, N, M}
    s = 0
    for node in PreOrderDFS(x -> abs(x.t - n.t) <= t, n)
        if isnothing(node.left)
            s += (t == node.t)
        else
            s += abs(node.t - n.t) <= t < abs(node.left.t - n.t)
        end
    end
    return s
end

"""
    timeshift!(n::ADSTaylorSolution{T, N, M}, dt::T) where {T <: Real, N, M}

Shift the time of every node in `n` by `dt`.
"""
function timeshift!(n::ADSTaylorSolution{T, N, M}, dt::T) where {T <: Real, N, M}
    for node in PreOrderDFS(n)
        node.t += dt
    end
end

"""
    evaltree(n, t [, s])

Evaluate binary tree `n` at time `t` (and domain point `s`).
"""
function evaltree(n::ADSTaylorSolution{T, N, M}, t::U) where {T<:Real, U<:Number, N, M}
    # t is outside time range
    if isnothing(n.left) || (n.t < n.left.t && t < n.t) ||
        (n.t > n.left.t && t > n.t)
        return Matrix{TaylorN{T}}(undef, 0, 0), Matrix{TaylorN{T}}(undef, 0, 0),
            Matrix{TaylorN{T}}(undef, 0, 0)
    end
    # Allocate set of nodes
    ns = Set{ADSTaylorSolution{T, N, M}}()
    # Search nodes at time t (recursively)
    for node in PreOrderDFS(x -> abs(x.t - n.t) <= t, n)
        if isnothing(node.left) && (t == node.t)
            push!(ns, node.parent)
        else
            if abs(node.t - n.t) <= t < abs(node.left.t - n.t)
                push!(ns, node)
            end
        end
    end
    # Set{...} to Vector{...}
    nodes = collect(ns)
    # Sort nodes by lowest corner of domain
    sort!(nodes, by = x -> x.lo)
    # Number of nodes at time t
    L = length(nodes)
    # There are 0 nodes at time t
    iszero(L) && return Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),
        Matrix{TaylorN{T}}(undef, 0, 0)
    # Domain of each node
    lo = Matrix{T}(undef, N, L)
    hi = Matrix{T}(undef, N, L)
    # State vector of each node
    x0 = Matrix{TaylorN{T}}(undef, M, L)
    # Evaluate polynomials at time t
    for i in eachindex(nodes)
        node = nodes[i]
        dt = t - node.t
        lo[:, i] .= node.lo
        hi[:, i] .= node.hi
        x0[:, i] .= zero.(node.x)
        evaluate!(node.p, dt, view(x0, :, i))
    end

    return lo, hi, x0
end

# Function-like callability method
(n::ADSTaylorSolution{T, N, M})(t::U) where {T <: Real, U <: Number, N, M} =
    evaltree(n, t)

function evaltree(n::ADSTaylorSolution{T, N, M}, t::U,
                  s::AbstractVector{T}) where {T <: Real, U <: Number, N, M}
    # Check length(s)
    @assert length(s) == N
    # t is outside time range or s is outside domain
    if !all(n.lo .<= s .<= n.hi) || isnothing(n.left) ||
        (n.t < n.left.t && t < n.t) || (n.t > n.left.t && t > n.t)
        return Vector{T}(undef, 0)
    end
    # Allocate set of nodes
    ns = Set{ADSTaylorSolution{T, N, M}}()
    # Search nodes at time t and domain s (recursively)
    for node in PreOrderDFS(x -> all(x.lo .<= s .<= x.hi) && abs(x.t - n.t) <= t, n)
        if isnothing(node.left) && (t == node.t)
            push!(ns, node.parent)
        else
            if abs(node.t - n.t) <= t < abs(node.left.t - n.t)
                push!(ns, node)
            end
        end
    end
    # Set{...} to Vector{...}
    nodes = collect(ns)
    # Number of nodes at time t and domain s
    L = length(nodes)
    # There are 0 nodes at time t and domain s
    # There are 0 nodes at time t
    iszero(L) && return Vector{T}(undef, 0)
    # State vector of each node
    x0 = Matrix{T}(undef, M, L)
    p = Matrix{TaylorN{T}}(undef, M, L)
    # Evaluate polynomials at time t and domain s
    for i in eachindex(nodes)
        node = nodes[i]
        dt = t - node.t
        # Evaluate node polynomial at dt
        p[:, i] .= zero.(node.x)
        evaluate!(node.p, dt, view(p, :, i))
        # Root node
        root = getroot(n)
        # Linear transformation
        ms = (root.hi - root.lo) ./ (node.hi - node.lo)
        ks = root.lo - node.lo .* ms
        z = Vector{T}(ms .* s .+ ks)
        # Eval p at transformed point
        evaluate!(p[:, i], z, view(x0, :, i))
    end
    # Average
    return sum(x0, dims = 2)[:] ./ L
end

# Function-like callability method
(n::ADSTaylorSolution{T, N, M})(t::U,
    s::AbstractVector{T}) where {T <: Real, U <: Number, N, M} = evaltree(n, t, s)

## ADS integrator

# Exponential model y(t) = A * exp(B * t) used to estimate the truncation error
# of jet transport polynomials
exp_model(t, p) = p[1] * exp(p[2] * t)
exp_model!(F, t, p) = (@. F = p[1] * exp(p[2] * t))

# In place jacobian of exp_model! wrt parameters p
function exp_model_jacobian!(J::Array{T, 2}, t, p) where {T <: Real}
    @. J[:, 1] = exp(p[2] * t)
    @. @views J[:, 2] = t * p[1] * J[:, 1]
end

"""
    truncerror(P::TaylorN{T}) where {T <: Real}
    truncerror(P::AbstractVector{TaylorN{T}}) where {T <: Real}

Return the truncation error of each jet transport variable in `P`.

!!! reference
    See section 3 of https://doi.org/10.1007/s10569-015-9618-3.
"""
function truncerror(P::TaylorN{T}) where {T <: Real}
    # Jet transport order
    varorder = P.order
    # Number of variables
    nv = get_numvars()
    # Absolute sum per variable per order
    ys = zeros(T, nv, varorder+1)

    for i in eachindex(P.coeffs)
        idxs = TaylorSeries.coeff_table[i]
        for (j, jdxs) in enumerate(idxs)
            coef = abs(P.coeffs[i].coeffs[j])
            for k in eachindex(jdxs)
                ys[k, jdxs[k]+1] += coef
            end
        end
    end
    # Initial parameters
    p0 = ones(T, 2)
    # Orders
    xs = 0:varorder

    M = Vector{T}(undef, nv)
    for i in eachindex(M)
        # Non zero coefficients
        idxs = findall(!iszero, view(ys, i, :))
        # Fit exponential model
        fit = curve_fit(exp_model!, exp_model_jacobian!, view(xs, idxs),
                        view(ys, i, idxs), p0; inplace = true)
        # Estimate next order coefficient
        exp_model!(view(M, i:i), varorder+1, fit.param)
    end

    return M
end

function truncerror(P::AbstractVector{TaylorN{T}}) where {T <: Real}
    # Number of variables
    nv = get_numvars()
    # Size per variable per element of P
    norms = Vector{T}(undef, nv)
    # Sum over elements of P
    norms .= sum(truncerror.(P[:]))
    return norms
end

"""
    splitdirection(P::TaylorN{T}) where {T <: Real}
    splitdirection(P::AbstractVector{TaylorN{T}}) where {T <: Real}

Return the index of the jet transport variable with the largest truncation error
according to [`truncerror`](@ref).
"""
function splitdirection(P::TaylorN{T}) where {T <: Real}
    # Size per variable
    M = truncerror(P)
    # Variable with maximum error
    return argmax(M)[1]
end

function splitdirection(P::AbstractVector{TaylorN{T}}) where {T <: Real}
    # Size per variable
    M = truncerror(P)
    # Variable with maximum error
    return argmax(M)[1]
end

"""
    adsnorm(P::TaylorN{T}) where {T <: Real}

Return the truncation error of `P`.
"""
function adsnorm(P::TaylorN{T}) where {T <: Real}
    # Jet transport order
    varorder = P.order
    # Absolute sum per order
    ys = norm.((P[i] for i in eachindex(P)), 1)
    # Initial parameters
    p0 = ones(T, 2)
    # Orders
    xs = 0:varorder
    # Non zero coefficients
    idxs = findall(!iszero, ys)
    # Fit exponential model
    fit = curve_fit(exp_model!, exp_model_jacobian!, view(xs, idxs),
                        view(ys, idxs), p0; inplace = true)
    # Estimate next order coefficient
    return exp_model(varorder+1, fit.param)
end

# Split [lo, hi] in half along direction i
function halve(lo::SVector{N, T}, hi::SVector{N, T}, i::Int) where {T <: Real, N}
    @assert 1 <= i <= N
    mid = (lo[i] + hi[i])/2
    a = SVector{N, T}(i == j ? mid : hi[j] for j in 1:N)
    b = SVector{N, T}(i == j ? mid : lo[j] for j in 1:N)
    return lo, a, b, hi
end

# Split node in half
# See section 3 of https://doi.org/10.1007/s10569-015-9618-3
function halve!(node::ADSTaylorSolution{T, N, M}, dt::T,
    x0::Vector{TaylorN{T}}) where {T, N, M}
    # Split direction
    j = splitdirection(x0)
    # Split domain
    lo1, hi1, lo2, hi2 = halve(node.lo, node.hi, j)
    # Jet transport variables
    v = get_variables()
    # Root node
    r = getroot(node)
    # Left half
    v[j] = v[j]/2 + r.lo[j]/2
    x1 = SVector{M, TaylorN{T}}(x0[i](v) for i in eachindex(x0))
    leftchild!(node, node.t + dt, lo1, hi1, x1)
    # Right half
    v .= get_variables()
    v[j] = v[j]/2 + r.hi[j]/2
    x2 = SVector{M, TaylorN{T}}(x0[i](v) for i in eachindex(x0))
    rightchild!(node, node.t + dt, lo2, hi2, x2)

    return nothing
end

"""
    decidesplit!(node::ADSTaylorSolution{T, N, M}, dt::T, x0::Vector{TaylorN{T}},
                 nsplits::Int, maxsplits::Int, stol::T) where {T, N, M}

Split `node` in half if any element of `x0` has an `adsnorm` greater than `stol`.
"""
function decidesplit!(node::ADSTaylorSolution{T, N, M}, dt::T, x0::Vector{TaylorN{T}},
    nsplits::Int, maxsplits::Int, stol::T) where {T, N, M}
    # Split criteria for each element of x
    mask = adsnorm.(x0)
    # Split
    if nsplits < maxsplits && any(mask .> stol)
        halve!(node, dt, x0)
        # Update number of splits
        nsplits += 1
    # No split
    else
        leftchild!(node, node.t + dt, node.lo, node.hi, SVector{M, TaylorN{T}}(x0))
    end

    return nsplits
end

@inline function set_psol!(::Val{true}, node::ADSTaylorSolution{T, N, M},
    x::Vector{Taylor1{TaylorN{T}}}) where {T <: Real, N, M}
    node.p = deepcopy.(x)
    return nothing
end

@inline function set_psol!(::Val{false}, node::ADSTaylorSolution{T, N, M},
    x::Vector{Taylor1{TaylorN{T}}}) where {T <: Real, N, M}
    node.p = nothing
    return nothing
end

function taylorinteg(f!, q0::ADSTaylorSolution{T, N, M}, t0::T, tmax::T, order::Int,
    stol::T, abstol::T, params = nothing; maxsplits::Int = 10, maxsteps::Int = 500,
    parse_eqs::Bool = true, dense::Bool = true) where {T <: Real, N, M}

    # Initialize the vector of Taylor1 expansions
    t = t0 + Taylor1( T, order )
    x = Array{Taylor1{TaylorN{T}}}(undef, M)
    dx = Array{Taylor1{TaylorN{T}}}(undef, M)
    @inbounds for i in eachindex(x)
        x[i] = Taylor1( q0.x[i], order )
        dx[i] = Taylor1( zero(q0.x[i]), order )
    end

    # Determine if specialized jetcoeffs! method exists
    parse_eqs, rv = _determine_parsing!(parse_eqs, f!, t, x, dx, params)

    return _taylorinteg!(Val(dense), f!, q0, t0, tmax, order, stol, abstol, rv,
        params; parse_eqs, maxsplits, maxsteps)
end

function _taylorinteg!(dense::Val{D}, f!, q0::ADSTaylorSolution{T, N, M}, t0::T,
    tmax::T, order::Int, stol::T, abstol::T, rv::RetAlloc{Taylor1{TaylorN{T}}},
    params = nothing; parse_eqs::Bool = true, maxsplits::Int = 10,
    maxsteps::Int = 500) where {T <: Real, D, N, M}

    # Allocation
    δt = Vector{T}(undef, maxsplits)
    t = Vector{Taylor1{T}}(undef, maxsplits)
    x0 = Matrix{TaylorN{T}}(undef, M, maxsplits)
    x = Matrix{Taylor1{TaylorN{T}}}(undef, M, maxsplits)
    dx = Matrix{Taylor1{TaylorN{T}}}(undef, M, maxsplits)
    xaux = Matrix{Taylor1{TaylorN{T}}}(undef, M, maxsplits)
    @inbounds for j in eachindex(t)
        δt[j] = zero(T)
        t[j] = t0 + Taylor1( T, order )
        @inbounds for i in axes(x0, 1)
            x0[i, j] = q0.x[i]
            x[i, j] = Taylor1( q0.x[i], order )
            dx[i, j] = Taylor1( zero(q0.x[i]), order )
            xaux[i, j] = Taylor1( zero(q0.x[i]), order )
        end
    end
    # IMPORTANT: each split needs its own RetAlloc
    rvs = [deepcopy(rv) for _ in 1:maxsplits]

    # Integration
    nsteps = 1
    nsplits = 1
    sign_tstep = copysign(1, tmax - t0)
    mask = BitVector(true for _ in 1:maxsplits)

    while any(view(mask, 1:nsplits))
        for (k, node) in enumerate(Leaves(q0))
            mask[k] = sign_tstep * node.t < sign_tstep * tmax
            mask[k] || continue
            @inbounds for i in axes(x, 1)
                x[i, k][0] = node.x[i]
                TaylorSeries.zero!(dx[i, k], 0)
            end
            δt[k] = taylorstep!(Val(parse_eqs), f!, t[k], x[:, k], dx[:, k], xaux[:, k],
                    abstol, params, rvs[k]) # δt is positive!
            # Below, δt has the proper sign according to the direction of the integration
            δt[k] = sign_tstep * min(δt[k], sign_tstep*(tmax-node.t))
            evaluate!(x[:, k], δt[k], view(x0, :, k)) # new initial condition
            set_psol!(dense, node, x[:, k]) # Store the Taylor polynomial solution
            @inbounds t[k][0] = node.t + δt[k]
            nsplits = decidesplit!(node, δt[k], x0[:, k], nsplits, maxsplits, stol)
        end
        nsteps += 1
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return nothing
end