# This file is part of the TaylorIntegration.jl package; MIT licensed

# set_psol!
@doc doc"""
    set_psol!(::Val{true}, psol::Array{Taylor1{U},1}, nsteps::Int, x::Taylor1{U}) where {U<:Number}
    set_psol!(::Val{true}, psol::Array{Taylor1{U},2}, nsteps::Int, x::Vector{Taylor1{U}}) where {U<:Number}
    set_psol!(::Val{false}, args...)

Auxiliary function to save Taylor polynomials in a call to [`taylorinteg`](@ref). When the
first argument in the call signature is `Val(true)`, sets appropriate elements of argument
`psol`. Otherwise, when the first argument in the call signature is `Val(false)`, this
function simply returns `nothing`. Argument `psol` is the array
where the Taylor polynomials associated to the solution will be stored, corresponding to
field `:p` in [`TaylorSolution`](@ref). See also [`init_psol`](@ref).
"""
@inline function set_psol!(::Val{true}, psol::Array{Taylor1{U},1}, nsteps::Int, x::Taylor1{U}) where {U<:Number}
    @inbounds psol[nsteps] = deepcopy(x)
    return nothing
end
@inline function set_psol!(::Val{true}, psol::Array{Taylor1{U},2}, nsteps::Int, x::Vector{Taylor1{U}}) where {U<:Number}
    @inbounds psol[:,nsteps] .= deepcopy.(x)
    return nothing
end
@inline set_psol!(::Val{false}, args...) = nothing

# init_psol
@doc doc"""
    init_psol(::Val{true}, maxsteps::Int, ::Int, ::Taylor1{U}) where {U<:Number}
    init_psol(::Val{true}, maxsteps::Int, dof::Int, ::Array{Taylor1{U},1}) where {U<:Number}
    init_psol(::Val{false}, ::Int, ::Int, ::Taylor1{U}) where {U<:Number}
    init_psol(::Val{false}, ::Int, ::Int, ::Array{Taylor1{U},1}) where {U<:Number}

Auxiliary function to initialize `psol` during a call to [`taylorinteg`](@ref). When the
first argument in the call signature is `Val(false)` this function simply returns `nothing`.
Otherwise, when the first argument in the call signature is `Val(true)`, then the
appropriate array is allocated and returned; this array is where the Taylor polynomials
associated to the solution will be stored, corresponding to field `:p` in
[`TaylorSolution`](@ref).
"""
@inline function init_psol(::Val{true}, maxsteps::Int, ::Int, ::Taylor1{U}) where {U<:Number}
    return Array{Taylor1{U}}(undef, maxsteps)
end
@inline function init_psol(::Val{true}, maxsteps::Int, dof::Int, ::Array{Taylor1{U},1}) where {U<:Number}
    return Array{Taylor1{U}}(undef, dof, maxsteps)
end

@inline init_psol(::Val{false}, ::Int, ::Int, ::Taylor1{U}) where {U<:Number} = nothing
@inline init_psol(::Val{false}, ::Int, ::Int, ::Array{Taylor1{U},1}) where {U<:Number} = nothing

# taylorinteg
function taylorinteg(f, x0::U, t0::T, tmax::T, order::Int, abstol::T, params = nothing;
        maxsteps::Int=500, parse_eqs::Bool=true, dense::Bool=true) where {T<:Real, U<:Number}

    # Allocation
    cache = init_cache(ScalarCache, Val(dense), t0, x0, maxsteps, order)

    # Determine if specialized jetcoeffs! method exists
    parse_eqs, rv = _determine_parsing!(parse_eqs, f, cache.t, cache.x, params)

    return taylorinteg!(Val(dense), f, x0, t0, tmax, abstol, rv, cache, params; parse_eqs, maxsteps)
end

function taylorinteg!(dense::Val{D}, f,
        x0::U, t0::T, tmax::T, abstol::T, rv::RetAlloc{Taylor1{U}}, cache::ScalarCache, params;
        parse_eqs::Bool=true, maxsteps::Int=500) where {T<:Real,U<:Number,D}

    @unpack tv, xv, psol, t, x = cache

    # Initial conditions
    nsteps = 1
    @inbounds t[0] = t0
    @inbounds tv[1] = t0
    @inbounds xv[1] = x0
    sign_tstep = copysign(1, tmax-t0)

    # Integration
    while sign_tstep*t0 < sign_tstep*tmax
        δt = taylorstep!(Val(parse_eqs), f, t, x, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep*(tmax-t0))
        x0 = evaluate(x, δt) # new initial condition
        set_psol!(dense, psol, nsteps, x) # Store the Taylor polynomial solution
        @inbounds x[0] = x0
        t0 += δt
        @inbounds t[0] = t0
        nsteps += 1
        @inbounds tv[nsteps] = t0
        @inbounds xv[nsteps] = x0
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return build_solution(tv, xv, psol, nsteps)
end


function taylorinteg(f!, q0::Vector{U}, t0::T, tmax::T, order::Int, abstol::T, params = nothing;
        maxsteps::Int=500, parse_eqs::Bool=true, dense::Bool=true) where {T<:Real, U<:Number}

    # Allocation
    cache = init_cache(VectorCache, Val(dense), t0, q0, maxsteps, order)

    # Determine if specialized jetcoeffs! method exists
    parse_eqs, rv = _determine_parsing!(parse_eqs, f!, cache.t, cache.x, cache.dx, params)

    return taylorinteg!(Val(dense), f!, q0, t0, tmax, abstol, rv,
        cache, params; parse_eqs, maxsteps)
end

function taylorinteg!(dense::Val{D}, f!,
        q0::Array{U,1}, t0::T, tmax::T, abstol::T, rv::RetAlloc{Taylor1{U}}, cache::VectorCache, params;
        parse_eqs::Bool=true, maxsteps::Int=500) where {T<:Real,U<:Number,D}

    @unpack tv, xv, psol, xaux, t, x, dx = cache

    # Initial conditions
    @inbounds t[0] = t0
    x0 = deepcopy(q0)
    @inbounds tv[1] = t0
    @inbounds xv[:,1] .= q0
    sign_tstep = copysign(1, tmax-t0)

    # Integration
    nsteps = 1
    while sign_tstep*t0 < sign_tstep*tmax
        δt = taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep*(tmax-t0))
        evaluate!(x, δt, x0) # new initial condition
        set_psol!(dense, psol, nsteps, x) # Store the Taylor polynomial solution
        @inbounds for i in eachindex(x0)
            x[i][0] = x0[i]
            TaylorSeries.zero!(dx[i], 0)
        end
        t0 += δt
        @inbounds t[0] = t0
        nsteps += 1
        @inbounds tv[nsteps] = t0
        @inbounds xv[:,nsteps] .= deepcopy.(x0)
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return build_solution(tv, xv, psol, nsteps)
end

@doc doc"""
    taylorinteg(f, x0, t0, tmax, order, abstol, params[=nothing]; kwargs... )

General-purpose Taylor integrator for the explicit ODE ``\dot{x}=f(x, p, t)``,
where `p` are the parameters encoded in `params`.
The initial conditions are specified by `x0` at time `t0`; `x0` may be of type `T<:Number`
or `Vector{T}`, with `T` including `TaylorN{T}`; the latter case
is of interest for [jet transport applications](@ref jettransport).

The equations of motion are specified by the function `f`; we follow the same
convention of `DifferentialEquations.jl` to define this function, i.e.,
`f(x, p, t)` or `f!(dx, x, p, t)`; see the examples below.

The functions returns a `TaylorSolution`, whose fields are `t` and `x`; they represent,
respectively, a vector with the values of time (independent variable),
and a vector with the computed values of
the dependent variable(s). When the keyword argument `dense` is set to `true`, it also
outputs in the field `p` the Taylor polynomial expansion computed at each time step.
The integration stops when time is larger than `tmax`, in which case the last returned
value(s) correspond to `tmax`, or when the number of saved steps is larger
than `maxsteps`.

The integration method uses polynomial expansions on the independent variable
of order `order`; the parameter `abstol` serves to define the
time step using the last two Taylor coefficients of the expansions.
Make sure you use a *large enough* `order` to assure convergence.

Currently, the recognized keyword arguments are:
- `maxsteps[=500]`: maximum number of integration steps.
- `parse_eqs[=true]`: use the specialized method of `jetcoeffs!` created
    with [`@taylorize`](@ref).
- `dense[=true]`: output the Taylor polynomial expansion at each time step.

## Examples

For one dependent variable the function `f` defines the RHS of the equation of
motion, returning the value of ``\dot{x}``. The arguments of
this function are `(x, p, t)`, where `x` are the dependent variables, `p` are
the paremeters and `t` is the independent variable.

For several (two or more) dependent variables, the function `f!` defines
the RHS of the equations of motion, mutating (in-place) the (preallocated) vector
with components of ``\dot{x}``. The arguments of this function are `(dx, x, p, t)`,
where `dx` is the preallocated vector of ``\dot{x}``, `x` are the dependent
variables, `p` are the paremeters entering the ODEs and `t` is the independent
variable. The function may return this vector or simply `nothing`.

```julia
using TaylorIntegration

f(x, p, t) = x^2

sol = taylorinteg(f, 3, 0.0, 0.3, 25, 1.0e-20, maxsteps=100 )

function f!(dx, x, p, t)
    for i in eachindex(x)
        dx[i] = x[i]^2
    end
    return nothing
end

sol = taylorinteg(f!, [3, 3], 0.0, 0.3, 25, 1.0e-20, maxsteps=100 )

sol = taylorinteg(f!, [3, 3], 0.0, 0.3, 25, 1.0e-20, maxsteps=100, dense=true )
```

""" taylorinteg


# Integrate and return results evaluated at given time
@doc doc"""
    taylorinteg(f, x0, trange, order, abstol, params[=nothing]; kwargs... )

General-purpose Taylor integrator for the explicit ODE
``\dot{x}=f(x,p,t)`` with initial condition specified by `x0::{T<:Number}`
or `x0::Vector{T}` at time `t0`.

The method returns a `TaylorSolution` whose field `x` represents the computed values of
the dependent variable(s), evaluated *only* at the times specified by
the range `trange`.

## Examples

```julia
sol = taylorinteg(f, 3, 0.0:0.001:0.3, 25, 1.0e-20, maxsteps=100 )

sol = taylorinteg(f!, [3, 3], 0.0:0.001:0.3, 25, 1.0e-20, maxsteps=100 );

```

"""
function taylorinteg(f, x0::U, trange::AbstractVector{T},
        order::Int, abstol::T, params = nothing;
        maxsteps::Int=500, parse_eqs::Bool=true) where {T<:Real, U<:Number}

    # Check if trange is increasingly or decreasingly sorted
    @assert (issorted(trange) ||
        issorted(trange, rev=true)) "`trange` or `reverse(trange)` must be sorted"

    # Allocation
    cache = init_cache(ScalarCache, Val(false), trange, x0, maxsteps, order)

    # Determine if specialized jetcoeffs! method exists
    parse_eqs, rv = _determine_parsing!(parse_eqs, f, cache.t, cache.x, params)

    return taylorinteg!(f, x0, trange, abstol, rv, cache, params; parse_eqs, maxsteps)
end

function taylorinteg!(f, x0::U, trange::AbstractVector{T},
        abstol::T, rv::RetAlloc{Taylor1{U}}, cache::ScalarCache, params; parse_eqs::Bool=true, maxsteps::Int=500) where {T<:Real, U<:Number}

    @unpack xv, t, x = cache

    # Initial conditions
    @inbounds t0, t1, tmax = trange[1], trange[2], trange[end]
    sign_tstep = copysign(1, tmax-t0)
    @inbounds t[0] = t0
    @inbounds xv[1] = x0

    # Integration
    iter = 2
    nsteps = 1
    while sign_tstep*t0 < sign_tstep*tmax
        δt = taylorstep!(Val(parse_eqs), f, t, x, abstol, params, rv)# δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep*(tmax-t0))
        x0 = evaluate(x, δt) # new initial condition
        tnext = t0+δt
        # Evaluate solution at times within convergence radius
        while sign_tstep*t1 < sign_tstep*tnext
            x1 = evaluate(x, t1-t0)
            @inbounds xv[iter] = x1
            iter += 1
            @inbounds t1 = trange[iter]
        end
        if δt == tmax-t0
            @inbounds xv[iter] = x0
            break
        end
        @inbounds x[0] = x0
        t0 = tnext
        @inbounds t[0] = t0
        nsteps += 1
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end
    return build_solution(trange, xv)
end

function taylorinteg(f!, q0::Vector{U}, trange::AbstractVector{T},
        order::Int, abstol::T, params = nothing;
        maxsteps::Int=500, parse_eqs::Bool=true) where {T<:Real, U<:Number}

    # Check if trange is increasingly or decreasingly sorted
    @assert (issorted(trange) ||
        issorted(trange, rev=true)) "`trange` or `reverse(trange)` must be sorted"

    # Allocation
    cache = init_cache(VectorTRangeCache, Val(false), trange, q0, maxsteps, order)

    # Determine if specialized jetcoeffs! method exists
    parse_eqs, rv = _determine_parsing!(parse_eqs, f!, cache.t, cache.x, cache.dx, params)

    return taylorinteg!(f!, q0, trange, abstol, rv,
        cache, params; parse_eqs, maxsteps)
end

function taylorinteg!(f!,
        q0::Vector{U}, trange::AbstractVector{T}, abstol::T, rv::RetAlloc{Taylor1{U}}, cache::VectorTRangeCache, params;
        parse_eqs::Bool=true, maxsteps::Int=500) where {T<:Real, U<:Number}

    @unpack xv, xaux, x0, x1, t, x, dx = cache

    # Initial conditions
    @inbounds t[0] = trange[1]
    @inbounds t0, t1, tmax = trange[1], trange[2], trange[end]
    sign_tstep = copysign(1, tmax-t0)
    # x .= Taylor1.(q0, order)
    @inbounds x0 .= deepcopy(q0)
    @inbounds xv[:,1] .= q0

    # Integration
    iter = 2
    nsteps = 1
    while sign_tstep*t0 < sign_tstep*tmax
        δt = taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep*(tmax-t0))
        evaluate!(x, δt, x0) # new initial condition
        tnext = t0+δt
        # Evaluate solution at times within convergence radius
        while sign_tstep*t1 < sign_tstep*tnext
            evaluate!(x, t1-t0, x1)
            @inbounds xv[:,iter] .= x1
            iter += 1
            @inbounds t1 = trange[iter]
        end
        if δt == tmax-t0
            @inbounds xv[:,iter] .= x0
            break
        end
        @inbounds for i in eachindex(x0)
            x[i][0] = x0[i]
            dx[i][0] = zero(x0[i])
        end
        t0 = tnext
        @inbounds t[0] = t0
        nsteps += 1
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return build_solution(trange, xv)
end


# Generic functions
for R in (:Number, :Integer)
    @eval begin

        function taylorinteg(f, xx0::S, tt0::T, ttmax::U, order::Int, aabstol::V,
                params = nothing; dense=false, maxsteps::Int=500, parse_eqs::Bool=true) where
                {S<:$R, T<:Real, U<:Real, V<:Real}

            # In order to handle mixed input types, we promote types before integrating:
            t0, tmax, abstol, _ = promote(tt0, ttmax, aabstol, one(Float64))
            x0, _ = promote(xx0, t0)

            return taylorinteg(f, x0, t0, tmax, order, abstol, params,
                dense=dense, maxsteps=maxsteps, parse_eqs=parse_eqs)
        end

        function taylorinteg(f, q0::Array{S,1}, tt0::T, ttmax::U, order::Int, aabstol::V,
                params = nothing; dense=false, maxsteps::Int=500, parse_eqs::Bool=true) where
                {S<:$R, T<:Real, U<:Real, V<:Real}

            #promote to common type before integrating:
            t0, tmax, abstol, _ = promote(tt0, ttmax, aabstol, one(Float64))
            elq0, _ = promote(q0[1], t0)
            #convert the elements of q0 to the common, promoted type:
            q0_ = convert(Array{typeof(elq0)}, q0)

            return taylorinteg(f, q0_, t0, tmax, order, abstol, params,
                dense=dense, maxsteps=maxsteps, parse_eqs=parse_eqs)
        end

        function taylorinteg(f, xx0::S, trange::AbstractVector{T}, order::Int, aabstol::U, params = nothing;
                maxsteps::Int=500, parse_eqs::Bool=true) where {S<:$R, T<:Real, U<:Real}

            t0, abstol, _ = promote(trange[1], aabstol, one(Float64))
            x0, _ = promote(xx0, t0)

            return taylorinteg(f, x0, trange .* one(t0), order, abstol, params,
                maxsteps=maxsteps, parse_eqs=parse_eqs)
        end

        function taylorinteg(f, q0::Array{S,1}, trange::AbstractVector{T}, order::Int, aabstol::U, params = nothing;
                maxsteps::Int=500, parse_eqs::Bool=true) where {S<:$R, T<:Real, U<:Real}

            t0, abstol, _ = promote(trange[1], aabstol, one(Float64))
            elq0, _ = promote(q0[1], t0)
            q0_ = convert(Array{typeof(elq0)}, q0)

            return taylorinteg(f, q0_, trange .* one(t0), order, abstol, params,
                maxsteps=maxsteps, parse_eqs=parse_eqs)
        end

    end
end
