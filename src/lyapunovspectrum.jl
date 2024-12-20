# This file is part of the TaylorIntegration.jl package; MIT licensed

# build_lyap_solution

@doc doc"""
    build_lyap_solution(t, x, λ, nsteps)
    build_lyap_solution(t, x, λ)

Helper function to build a [`TaylorSolution`](@ref) from a call to
[`lyap_taylorinteg`](@ref).

"""
build_lyap_solution(t::AbstractVector{T},
        x::Matrix{U},
        λ::Matrix{U},
        nsteps::Int) where {T, U} =
    TaylorSolution(arraysol(t, nsteps),
        arraysol(x, nsteps),
        nothing,
        nothing,
        nothing,
        nothing,
        arraysol(λ, nsteps))

build_lyap_solution(t::AbstractVector{T},
        x::Matrix{U},
        λ::Matrix{U}) where {T, U} =
    TaylorSolution(t,
        transpose(x),
        nothing,
        nothing,
        nothing,
        nothing,
        transpose(λ))

# stabilitymatrix!

"""
    stabilitymatrix!(eqsdiff!, t, x, δx, dδx, jac, dvars, params[, jacobianfunc!=nothing])

Updates the matrix `jac::Matrix{Taylor1{U}}` (linearized equations of motion)
computed from the equations of motion (`eqsdiff!`), at time `t` at `x`; `x` is
of type `Vector{Taylor1{U}}`, where `U<:Number`. `δx`, `dδx` and `dvars` are
auxiliary arrays of type `Vector{TaylorN{Taylor1{U}}}` to avoid allocations.
Optionally, the user may provide a Jacobian function `jacobianfunc!` to compute
`jac`. Otherwise, `jac` is computed via automatic differentiation using
`TaylorSeries.jl`.

"""
function stabilitymatrix!(eqsdiff!, t::Taylor1{T}, x::Vector{Taylor1{U}},
        δx::Vector{TaylorN{Taylor1{U}}}, dδx::Vector{TaylorN{Taylor1{U}}},
        jac::Matrix{Taylor1{U}}, dvars::Vector{TaylorN{Taylor1{U}}}, params,
        ::Nothing) where {T<:Real, U<:Number}
    # Set δx equal to current value of x plus 1st-order variations
    for ind in eachindex(δx)
        @inbounds δx[ind] = x[ind] + dvars[ind]
    end
    # Equations of motion
    eqsdiff!(dδx, δx, params, t)
    TaylorSeries.jacobian!(jac, dδx)
    nothing
end
function stabilitymatrix!(eqsdiff!, t::Taylor1{T}, x::Vector{Taylor1{U}},
        δx::Vector{TaylorN{Taylor1{U}}}, dδx::Vector{TaylorN{Taylor1{U}}},
        jac::Matrix{Taylor1{U}}, dvars::Vector{TaylorN{Taylor1{U}}}, params,
        jacobianfunc!::J) where {T<:Real, U<:Number, J}
    jacobianfunc!(jac, x, params, t)
    nothing
end
stabilitymatrix!(eqsdiff!, t, x, δx, dδx, jac, dvars, params) =
    stabilitymatrix!(eqsdiff!, t, x, δx, dδx, jac, dvars, params, nothing)

# Modified from `cgs` and `mgs`, obtained from:
# http://nbviewer.jupyter.org/url/math.mit.edu/~stevenj/18.335/Gram-Schmidt.ipynb
# Classical Gram–Schmidt (Trefethen algorithm 7.1), implemented in the simplest way
# (We could make it faster by unrolling loops to avoid temporaries arrays etc.)
function classicalGS!(A, Q, R, aⱼ, qᵢ, vⱼ)
    m,n = size(A)
    fill!(R, zero(eltype(A)))
    for j = 1:n
        # aⱼ = A[:,j]
        for ind = 1:m
            @inbounds aⱼ[ind] = A[ind,j]
            @inbounds vⱼ[ind] = aⱼ[ind]
        end
        # vⱼ = copy(aⱼ) # use copy so that modifying vⱼ doesn't change aⱼ
        for i = 1:j-1
            # qᵢ = Q[:,i]
            for ind = 1:m
                @inbounds qᵢ[ind] = Q[ind,i]
            end
            @inbounds R[i,j] = dot(qᵢ, aⱼ)
            # vⱼ -= R[i,j] * qᵢ
            @inbounds for ind = 1:m
                vⱼ[ind] -= R[i,j] * qᵢ[ind]
            end
        end
        @inbounds R[j,j] = norm(vⱼ)
        # Q[:,j] = vⱼ / R[j,j]
        for ind = 1:m
            @inbounds Q[ind,j] = vⱼ[ind] / R[j,j]
        end
    end
    return nothing
end

# Modified Gram–Schmidt (Trefethen algorithm 8.1); see also
# http://nbviewer.jupyter.org/url/math.mit.edu/~stevenj/18.335/Gram-Schmidt.ipynb
function modifiedGS!(A, Q, R, aⱼ, qᵢ, vⱼ)
    m,n = size(A)
    fill!(R, zero(eltype(A)))
    for j = 1:n
        # aⱼ = A[:,j]
        for ind = 1:m
            @inbounds aⱼ[ind] = A[ind,j]
            @inbounds vⱼ[ind] = aⱼ[ind]
        end
        # vⱼ = copy(aⱼ) # use copy so that modifying vⱼ doesn't change aⱼ
        for i = 1:j-1
            # qᵢ = Q[:,i]
            for ind = 1:m
                @inbounds qᵢ[ind] = Q[ind,i]
            end
            @inbounds R[i,j] = dot(qᵢ, vⱼ) # ⟵ NOTICE: mgs has vⱼ, clgs has aⱼ
            # vⱼ -= R[i,j] * qᵢ
            @inbounds for ind = 1:m
                vⱼ[ind] -= R[i,j] * qᵢ[ind]
            end
        end
        @inbounds R[j,j] = norm(vⱼ)
        # Q[:,j] = vⱼ / R[j,j]
        for ind = 1:m
            @inbounds Q[ind,j] = vⱼ[ind] / R[j,j]
        end
    end
    return nothing
end

@doc doc"""
    lyap_jetcoeffs!(t, x, dx, jac, varsaux)

Similar to [`jetcoeffs!`](@ref) for the calculation of the Lyapunov spectrum.
Updates *only* the elements of `x` which correspond to the solution of the 1st-order
variational equations ``\dot{\xi}=J \cdot \xi``, where ``J`` is the Jacobian
matrix, i.e., the linearization of the equations of motion. `jac` is the Taylor
expansion of ``J`` wrt the independent variable, around the current initial
condition. `varsaux` is an auxiliary array of type `Array{eltype(jac),3}` to
avoid allocations. Calling this method assumes that `jac` has been computed
previously using [`stabilitymatrix!`](@ref).

"""
function lyap_jetcoeffs!(t::Taylor1{T}, x::AbstractVector{Taylor1{S}},
        dx::AbstractVector{Taylor1{S}}, jac::Matrix{Taylor1{S}},
        varsaux::Array{Taylor1{S},3}) where {T <: Real, S <: Number}
    order = t.order
    # `dofrange` behaves like 1:dof, where `dof = size(jac, 1)`. Used to initialize `b` and `li`
    dofrange = axes(jac, 1)
    # The `view` below allows us to obtain CartesianIndices from `varsaux` when calling `eachindex(b)`
    # `varsaux` is an array with size n×n×n
    b = view(varsaux, dofrange, dofrange, dofrange)
    # We use `li` to map between cartesian and linear indices
    li = LinearIndices((dofrange, dofrange))

    # 0-th order evaluation of variational equations `dx = jac * x`
    # Initialize LHS of variational equations at zero
    dx .= zero(x[1])
    # Compute 0-th Taylor coefficients of matrix product `jac * x` and save into `dx`
    for i in eachindex(b)
        varsaux[i] = Taylor1(constant_term(jac[ i[1], i[3] ]) * constant_term(x[ li[i[3], i[2]] ]), order)
        lin_indx = li[ i[1], i[2]] # map from cartesian to linear index
        dx[lin_indx] = Taylor1(constant_term(dx[lin_indx]) + constant_term(varsaux[i]), order)
    end
    # Recursion relations, 0-th order
    for __idx = eachindex(x)
        (x[__idx]).coeffs[2] = (dx[__idx]).coeffs[1]
    end

    # Compute Taylor coefficients of variational equations `dx = jac * x` up to order `order`
    for ord = 1:order - 1
        ordnext = ord + 1
        # Compute `ord`-th Taylor coefficients of matrix product `jac * x` and save into `dx`
        for i in eachindex(b)
            TaylorSeries.mul!(varsaux[i], jac[ i[1], i[3] ], x[ li[i[3], i[2]] ], ord)
            lin_indx = li[ i[1], i[2]] # map from cartesian to linear index
            TaylorSeries.add!(dx[lin_indx], dx[lin_indx], varsaux[i], ord)
        end
        # Recursion relations, `ord`-th order
        for __idx = eachindex(x)
            (x[__idx]).coeffs[ordnext + 1] = (dx[__idx]).coeffs[ordnext] / ordnext
        end
    end

    return nothing
end

"""
    lyap_taylorstep!(::Val{V}, f!, t, x, dx, xaux, δx, dδx, jac, abstol, dvars, varsaux, params, rv[, jacobianfunc!])

Similar to [`taylorstep!`](@ref) for the calculation of the Lyapunov spectrum.
`jac` is the Taylor expansion (wrt the independent variable) of the
linearization of the equations of motion, i.e, the Jacobian. `xaux`, `δx`, `dδx`,
`varsaux` and `dvars` are auxiliary vectors, and `params` define the parameters
of the ODEs. Optionally, the user may provide a
Jacobian function `jacobianfunc!` to compute `jac`. Otherwise, `jac` is computed
via automatic differentiation using `TaylorSeries.jl`.

"""

function lyap_taylorstep!(::Val{V}, f!, t::Taylor1{T}, x::Vector{Taylor1{U}}, dx::Vector{Taylor1{U}},
    xaux::Vector{Taylor1{U}}, δx::Array{TaylorN{Taylor1{U}},1}, dδx::Array{TaylorN{Taylor1{U}},1},
    jac::Array{Taylor1{U},2}, abstol::T, dvars::Vector{TaylorN{Taylor1{U}}},
    varsaux::Array{Taylor1{U},3}, params, rv::RetAlloc{Taylor1{U}},
    jacobianfunc! =nothing) where {T<:Real, U<:Number, V}

    # Dimensions of phase-space: dof
    nx = length(x)
    dof = length(δx)

    # Compute the Taylor coefficients associated to trajectory
    __jetcoeffs!(Val(V), f!, t, view(x, 1:dof), view(dx, 1:dof), view(xaux, 1:dof), params, rv)

    # Compute stability matrix
    stabilitymatrix!(f!, t, x, δx, dδx, jac, dvars, params, jacobianfunc!)

    # Compute the Taylor coefficients associated to variational equations
    lyap_jetcoeffs!(t, view(x, dof+1:nx), view(dx, dof+1:nx), jac, varsaux)

    # Compute the step-size of the integration using `abstol`
    δt = stepsize(view(x, 1:dof), abstol)

    return δt
end

"""
    lyap_taylorinteg(f!, q0, t0, tmax, order, abstol[, jacobianfunc!=nothing];
        maxsteps::Int=500, parse_eqs::Bool=true)
    lyap_taylorinteg(f!, q0, trange, order, abstol[, jacobianfunc!=nothing];
        maxsteps::Int=500, parse_eqs::Bool=true)

Similar to [`taylorinteg`](@ref) for the calculation of the Lyapunov
spectrum. Note that the number of `TaylorN` variables should be set
previously by the user (e.g., by means of `TaylorSeries.set_variables`) and
should be equal to the length of the vector of initial conditions `q0`.
Otherwise, whenever `length(q0) != TaylorSeries.get_numvars()`, then
`lyap_taylorinteg` throws an `AssertionError`. Optionally, the user may provide
a Jacobian function `jacobianfunc!` to evaluate the current value of the Jacobian.
Otherwise, the current value of the Jacobian is computed via automatic
differentiation using `TaylorSeries.jl`.

"""
function lyap_taylorinteg(f!, q0::Array{U,1}, t0::T, tmax::T,
        order::Int, abstol::T, params = nothing, jacobianfunc! =nothing;
        maxsteps::Int=500, parse_eqs::Bool=true) where {T<:Real, U<:Number}

    dof = length(q0)

    # Allocation
    cache = init_cache(LyapunovSpectrumCache, Val(false), t0, q0, maxsteps, order)

    # If user does not provide Jacobian, check number of TaylorN variables and initialize _dv
    if isa(jacobianfunc!, Nothing)
        @assert get_numvars() == dof "`length(q0)` must be equal to number of variables set by `TaylorN`"
        for ind in eachindex(q0)
            cache.dvars[ind] = one(cache.x[1])*TaylorN(Taylor1{U}, ind, order=1)
        end
    end

    # Determine if specialized jetcoeffs! method exists
    parse_eqs, rv = _determine_parsing!(parse_eqs, f!, cache.t,
                    view(cache.x, 1:dof), view(cache.dx, 1:dof), params)

    return lyap_taylorinteg!(f!, q0, t0, tmax, abstol, rv,
        cache, params, jacobianfunc!; maxsteps, parse_eqs)
end

function lyap_taylorinteg!(f!,
        q0::Array{U,1}, t0::T, tmax::T, abstol::T,
        rv::RetAlloc{Taylor1{U}}, cache::LyapunovSpectrumCache, params, jacobianfunc!; parse_eqs::Bool=true,
        maxsteps::Int=500) where {T<:Real, U<:Number}

    @unpack tv, xv, xaux, x0, λ, λtsum, δx, dδx, jac, varsaux, QH, RH, aⱼ, qᵢ, vⱼ, t, x, dx, jt, dvars = cache

    # Allocation
    order = get_order(t)
    dof = length(q0)

    # Initial conditions
    @inbounds t[0] = t0
    sign_tstep = copysign(1, tmax-t0)
    t00 = t0
    tspan = zero(T)
    @inbounds tv[1] = t0
    @inbounds for ind in eachindex(q0)
        xv[ind,1] = q0[ind]
        λ[ind,1] = zero(U)
        λtsum[ind] = zero(U)
    end

    # Integration
    nsteps = 1
    while sign_tstep*t0 < sign_tstep*tmax
        δt = lyap_taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, δx, dδx, jac, abstol, dvars,
            varsaux, params, rv, jacobianfunc!) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep*(tmax-t0))
        evaluate!(x, δt, x0) # Update x0
        for ind in eachindex(jt)
            @inbounds jt[ind] = x0[dof+ind]
        end
        modifiedGS!( jt, QH, RH, aⱼ, qᵢ, vⱼ )
        t0 += δt
        @inbounds t[0] = t0
        tspan = t0-t00
        nsteps += 1
        @inbounds tv[nsteps] = t0
        @inbounds for ind in eachindex(q0)
            xv[ind,nsteps] = x0[ind]
            λtsum[ind] += log(RH[ind,ind])
            λ[ind,nsteps] = λtsum[ind]/tspan
        end
        for ind in eachindex(QH)
            @inbounds x0[dof+ind] = QH[ind]
        end
        @inbounds for i in eachindex(x0)
            x[i][0] = x0[i]
            TaylorSeries.zero!(dx[i], 0)
        end
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return build_lyap_solution(tv, xv, λ, nsteps)
end


function lyap_taylorinteg(f!, q0::Array{U,1}, trange::AbstractVector{T},
        order::Int, abstol::T, params = nothing, jacobianfunc! = nothing;
        maxsteps::Int=500, parse_eqs::Bool=true) where {T<:Real, U<:Number}

    dof = length(q0)

        # Check if trange is increasingly or decreasingly sorted
    @assert (issorted(trange) ||
        issorted(trange, rev=true)) "`trange` or `reverse(trange)` must be sorted"

    # Allocation
    cache = init_cache(LyapunovSpectrumTRangeCache, Val(false), trange, q0, maxsteps, order)

    # If user does not provide Jacobian, check number of TaylorN variables and initialize _dv
    if isnothing(jacobianfunc!)
        @assert get_numvars() == dof "`length(q0)` must be equal to number of variables set by `TaylorN`"
        for ind in eachindex(q0)
            cache.dvars[ind] = one(cache.x[1])*TaylorN(Taylor1{U}, ind, order=1)
        end
    end

    # Determine if specialized jetcoeffs! method exists
    parse_eqs, rv = _determine_parsing!(parse_eqs, f!, cache.t,
                    view(cache.x, 1:dof), view(cache.dx, 1:dof), params)

    return lyap_taylorinteg!(f!, q0, trange, abstol, rv, cache,
        params, jacobianfunc!; parse_eqs, maxsteps)

end

function lyap_taylorinteg!(f!,
        q0::Array{U,1}, trange::AbstractVector{T}, abstol::T,
        rv::RetAlloc{Taylor1{U}}, cache::LyapunovSpectrumTRangeCache, params, jacobianfunc!; parse_eqs::Bool=true,
        maxsteps::Int=500) where {T<:Real, U<:Number}

    @unpack xv, xaux, x0, q1, λ, λtsum, δx, dδx, jac, varsaux, QH, RH, aⱼ, qᵢ, vⱼ, t, x, dx, jt, dvars = cache

    # Auxiliary variables
    order = get_order(t)
    dof = length(q0)

    # Initial conditions
    @inbounds t[0] = trange[1]
    @inbounds t0, t1, tmax = trange[1], trange[2], trange[end]
    sign_tstep = copysign(1, tmax-t0)
    t00 = t0
    tspan = zero(T)
    @inbounds for ind in eachindex(q0)
        xv[ind,1] = q0[ind]
        λ[ind,1] = zero(U)
        λtsum[ind] = zero(U)
    end

    # Integration
    iter = 2
    nsteps = 1
    while sign_tstep*t0 < sign_tstep*tmax
        δt = lyap_taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, δx, dδx, jac, abstol, dvars,
            varsaux, params, rv, jacobianfunc!) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep*(tmax-t0))
        evaluate!(x, δt, x0) # Update x0
        tnext = t0+δt
        # Evaluate solution at times within convergence radius
        while t1 < tnext
            evaluate!(x[1:dof], t1-t0, q1)
            @inbounds xv[:,iter] .= q1
            for ind in eachindex(jt)
                @inbounds jt[ind] = evaluate(x[dof+ind], δt)
            end
            modifiedGS!( jt, QH, RH, aⱼ, qᵢ, vⱼ )
            tspan = t1-t00
            @inbounds for ind in eachindex(q0)
                λ[ind,iter] = (λtsum[ind]+log(RH[ind,ind]))/tspan
            end
            iter += 1
            @inbounds t1 = trange[iter]
        end
        if δt == tmax-t0
            @inbounds xv[:,iter] .= x0[1:dof]
            for ind in eachindex(jt)
                @inbounds jt[ind] = x0[dof+ind]
            end
            modifiedGS!( jt, QH, RH, aⱼ, qᵢ, vⱼ )
            tspan = tmax-t00
            @inbounds for ind in eachindex(q0)
                λ[ind,iter] = (λtsum[ind]+log(RH[ind,ind]))/tspan
            end
            break
        end

        for ind in eachindex(jt)
            @inbounds jt[ind] = x0[dof+ind]
        end
        modifiedGS!( jt, QH, RH, aⱼ, qᵢ, vⱼ )

        t0 = tnext
        @inbounds t[0] = t0
        nsteps += 1
        @inbounds for ind in eachindex(q0)
            λtsum[ind] += log(RH[ind,ind])
        end
        for ind in eachindex(QH)
            @inbounds x0[dof+ind] = QH[ind]
        end
        @inbounds for i in eachindex(x0)
            x[i][0] = x0[i]
            TaylorSeries.zero!(dx[i], 0)
        end
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return build_lyap_solution(trange, xv, λ)
end
