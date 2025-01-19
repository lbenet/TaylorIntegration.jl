# This file is part of the TaylorIntegration.jl package; MIT licensed

module TaylorIntegrationDiffEqExt

using TaylorIntegration

if isdefined(Base, :get_extension)
    using OrdinaryDiffEq: @unpack, @cache, OrdinaryDiffEqAdaptiveAlgorithm,
        OrdinaryDiffEqConstantCache, OrdinaryDiffEqMutableCache, ODEFunction,
        DynamicalODEFunction, check_keywords, warn_compat, ODEProblem, DynamicalODEProblem
    import OrdinaryDiffEq
else
    using ..OrdinaryDiffEq: @unpack, @cache, OrdinaryDiffEqAdaptiveAlgorithm,
        OrdinaryDiffEqConstantCache, OrdinaryDiffEqMutableCache, ODEFunction,
        DynamicalODEFunction, check_keywords, warn_compat, ODEProblem, DynamicalODEProblem
    import ..OrdinaryDiffEq
end

using StaticArrays: SVector, SizedArray
using RecursiveArrayTools: ArrayPartition, copyat_or_push!

import DiffEqBase

# TODO: check which keywords work fine
const warnkeywords = (:save_idxs, :d_discontinuities, :unstable_check, :save_everystep,
    :save_end, :initialize_save, :adaptive, :dt, :reltol, :dtmax,
    :dtmin, :force_dtmin, :internalnorm, :gamma, :beta1, :beta2,
    :qmax, :qmin, :qsteady_min, :qsteady_max, :qoldinit, :failfactor,
    :isoutofdomain, :unstable_check,
    :calck, :progress, :timeseries_steps)

global warnlist = Set(warnkeywords)



abstract type TaylorAlgorithm <: OrdinaryDiffEqAdaptiveAlgorithm end
struct TaylorMethodParams <: TaylorAlgorithm
    order::Int
    parse_eqs::Bool
end

using TaylorIntegration: RetAlloc
import TaylorIntegration: TaylorMethod, update_jetcoeffs_cache!

TaylorMethod(order; parse_eqs=true) = TaylorMethodParams(order, parse_eqs) # set `parse_eqs` to `true` by default

OrdinaryDiffEq.alg_order(alg::TaylorMethodParams) = alg.order

TaylorMethod() = error("Maximum order must be specified for the Taylor method")

# overload DiffEqBase.ODE_DEFAULT_NORM for Taylor1 arrays
DiffEqBase.ODE_DEFAULT_NORM(x::AbstractArray{<:AbstractSeries}, y) = norm(x, Inf)

### cache stuff
struct TaylorMethodCache{uType, rateType, tTType, uTType} <: OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType
    k::rateType
    fsalfirst::rateType
    tT::tTType
    uT::uTType
    duT::uTType
    uauxT::uTType
    parse_eqs::Ref{Bool}
    rv::RetAlloc
end

struct TaylorMethodConstantCache{uTType} <: OrdinaryDiffEqConstantCache
    uT::uTType
    parse_eqs::Ref{Bool}
    rv::RetAlloc{uTType}
end

# NOTE: This de-facto implies support Julia from v1.10 onwards
if VERSION >= v"1.10"
    OrdinaryDiffEq.get_fsalfirstlast(cache::TaylorMethodCache, u) = (cache.fsalfirst, cache.k)
end

function OrdinaryDiffEq.alg_cache(alg::TaylorMethodParams, u, rate_prototype, uEltypeNoUnits,
        uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p,
        calck,::Val{true})
    order = alg.order
    tT = Taylor1(typeof(t), order)
    tT[0] = t
    uT = Taylor1.(u, order)
    duT = zero.(Taylor1.(u, order))
    uauxT = similar(uT)
    parse_eqs, rv = TaylorIntegration._determine_parsing!(alg.parse_eqs, f, tT, uT, duT, p)
    return TaylorMethodCache(
        u,
        uprev,
        similar(u),
        zero(rate_prototype),
        zero(rate_prototype),
        tT,
        uT,
        duT,
        uauxT,
        Ref(parse_eqs),
        rv
        )
end

# This method is used for DynamicalODEFunction's (`parse_eqs=false`): tmpT1 and arrT1
# must have the proper type to build `TaylorMethodCache`
function OrdinaryDiffEq.alg_cache(alg::TaylorMethodParams, u::ArrayPartition, rate_prototype, uEltypeNoUnits,
        uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p,
        calck, ::Val{true})
    order = alg.order
    tT = Taylor1(typeof(t), order)
    tT[0] = t
    uT = Taylor1.(u, order)
    duT = zero.(Taylor1.(u, order))
    uauxT = similar(uT)
    parse_eqs, rv = TaylorIntegration._determine_parsing!(alg.parse_eqs, f, tT, uT, duT, p)
    return TaylorMethodCache(
        u,
        uprev,
        similar(u),
        zero(rate_prototype),
        zero(rate_prototype),
        tT,
        uT,
        duT,
        uauxT,
        Ref(parse_eqs),
        rv
        )
end

function OrdinaryDiffEq.alg_cache(alg::TaylorMethodParams, u, rate_prototype, uEltypeNoUnits,
        uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck,
        ::Val{false})
    order = alg.order
    tT = Taylor1(typeof(t), order)
    tT[0] = t
    uT = Taylor1(u, order)
    parse_eqs, rv = TaylorIntegration._determine_parsing!(alg.parse_eqs, f, tT, uT, p)
    return TaylorMethodConstantCache(Taylor1(u, alg.order), Ref(parse_eqs), rv)
end

function OrdinaryDiffEq.initialize!(integrator, c::TaylorMethodConstantCache)
    @unpack u, t, f, p = integrator
    tT = Taylor1(typeof(t), integrator.alg.order)
    tT[0] = t
    c.uT .= Taylor1(u, tT.order)
    TaylorIntegration.__jetcoeffs!(Val(c.parse_eqs.x), f, tT, c.uT, p, c.rv)
    # FSAL stuff
    integrator.kshortsize = 2
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)
    integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
    integrator.stats.nf += 1
    # Avoid undefined entries if k is an array of arrays
    integrator.fsallast = zero(integrator.fsalfirst)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
end

function OrdinaryDiffEq.perform_step!(integrator,cache::TaylorMethodConstantCache)
    @unpack u, t, dt, f, p = integrator
    tT = Taylor1(typeof(t), integrator.alg.order)
    tT[0] = t+dt
    u = evaluate(cache.uT, dt)
    cache.uT[0] = u
    TaylorIntegration.__jetcoeffs!(Val(cache.parse_eqs.x), f, tT, cache.uT, p, cache.rv)
    k = f(u, p, t+dt) # For the interpolation, needs k at the updated point
    integrator.stats.nf += 1
    integrator.fsallast = k
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.u = u
end

function OrdinaryDiffEq.initialize!(integrator, cache::TaylorMethodCache)
    @unpack u, t, f, p = integrator
    @unpack k, fsalfirst, tT, uT, duT, uauxT, parse_eqs, rv = cache
    TaylorIntegration.__jetcoeffs!(Val(parse_eqs.x), f, tT, uT, duT, uauxT, p, rv)
    # FSAL for interpolation
    integrator.fsalfirst = fsalfirst
    integrator.fsallast = k
    integrator.kshortsize = 1
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
    # integrator.f(integrator.fsalfirst,integrator.uprev,integrator.p,integrator.t)
    integrator.fsalfirst = constant_term.(duT)
    integrator.stats.nf += 1
end

function OrdinaryDiffEq.perform_step!(integrator, cache::TaylorMethodCache)
    @unpack t, dt, u, f, p = integrator
    @unpack k, tT, uT, duT, uauxT, parse_eqs, rv = cache
    evaluate!(uT, dt, u)
    tT[0] = t+dt
    for i in eachindex(u)
        @inbounds uT[i][0] = u[i]
        @inbounds TaylorSeries.zero!(duT[i], 0)
    end
    TaylorIntegration.__jetcoeffs!(Val(parse_eqs.x), f, tT, uT, duT, uauxT, p, rv)
    k = constant_term.(duT) # For the interpolation, needs k at the updated point
    integrator.stats.nf += 1
end

OrdinaryDiffEq.get_fsalfirstlast(cache::TaylorMethodCache, u) = (cache.fsalfirst, cache.k)

OrdinaryDiffEq.stepsize_controller!(integrator,alg::TaylorMethodParams) =
    TaylorIntegration.stepsize(integrator.cache.uT, integrator.opts.abstol)
OrdinaryDiffEq.step_accept_controller!(integrator, alg::TaylorMethodParams, q) = q

function DiffEqBase.solve(
        prob::DiffEqBase.AbstractODEProblem{uType, tupType, isinplace},
        alg::TaylorMethodParams, args...;
        verbose=true,
        kwargs...) where
        {uType, tupType, isinplace}

    # SciMLBase.unwrapped_f(prob.f)

    if verbose
        warned = !isempty(kwargs) && check_keywords(alg, kwargs, warnlist)
        warned && warn_compat()
    end

    f = prob.f
    parse_eqs = haskey(kwargs, :parse_eqs) ? kwargs[:parse_eqs] : true # `true` is the default
    if !isinplace && typeof(prob.u0) <: AbstractArray
        ### TODO: allow `parse_eqs=true` for DynamicalODEFunction
        if prob.f isa DynamicalODEFunction
            f1! = (dv, v, u, p, t) -> (dv .= prob.f.f1(v, u, p, t); 0)
            f2! = (du, v, u, p, t) -> (du .= prob.f.f2(v, u, p, t); 0)
            _alg = TaylorMethod(alg.order, parse_eqs = false)
            ### workaround use of `SVector` with oop `DynamicalODEProblem`
            ### TODO: add proper support for oop problems with arrays
            if eltype(prob.u0.x) <: SVector
                _u0 = ArrayPartition(SizedArray{Tuple{length(prob.u0.x[1])}}.(prob.u0.x))
            else
                _u0 = prob.u0
            end
            _prob = DynamicalODEProblem(f1!, f2!, _u0.x[1], _u0.x[2], prob.tspan, prob.p; prob.kwargs...)
        else
            f! = (du, u, p, t) -> (du .= f(u, p, t); 0)
            _alg = TaylorMethod(alg.order; parse_eqs)
            _prob = ODEProblem(f!, prob.u0, prob.tspan, prob.p; prob.kwargs...)
        end
    else
        _alg = TaylorMethod(alg.order; parse_eqs)
        _prob = prob
    end

    # DiffEqBase.solve(prob, _alg, args...; kwargs...)
    integrator = DiffEqBase.__init(_prob, _alg, args...; kwargs...)
    integrator.dt = integrator.tdir * TaylorIntegration.stepsize(integrator.cache.uT, integrator.opts.abstol) # override handle_dt! setting of initial dt
    DiffEqBase.solve!(integrator)
    integrator.sol
end

# used in continuous callbacks and related methods to update Taylor expansions cache
function update_jetcoeffs_cache!(u,f,p,cache::TaylorMethodCache)
    @unpack tT, uT, duT, uauxT, parse_eqs, rv = cache
    @inbounds for i in eachindex(u)
        @inbounds uT[i][0] = u[i]
        @inbounds TaylorSeries.zero!(duT[i], 0)
    end
    TaylorIntegration.__jetcoeffs!(Val(parse_eqs.x), f, tT, uT, duT, uauxT, p, rv)
    return nothing
end

# This function was modified from OrdinaryDiffEq.jl; MIT-licensed
# _ode_addsteps! overload for ::TaylorMethodCache to handle continuous
# and vector callbacks with TaylorIntegration.jl via the common interface
function OrdinaryDiffEq._ode_addsteps!(k, t, uprev, u, dt, f, p, cache::TaylorMethodCache,
        always_calc_begin = false, allow_calc_end = true,force_calc_end = false)
    ### TODO: check, and if necessary, reset timestep after callback (!)
    if length(k)<2 || always_calc_begin
        if typeof(cache) <: OrdinaryDiffEqMutableCache
            rtmp = similar(u, eltype(eltype(k)))
            f(rtmp,uprev,p,t)
            copyat_or_push!(k,1,rtmp)
            f(rtmp,u,p,t+dt)
            copyat_or_push!(k,2,rtmp)
        else
            copyat_or_push!(k,1,f(uprev,p,t))
            copyat_or_push!(k,2,f(u,p,t+dt))
        end
    end
    update_jetcoeffs_cache!(u,f,p,cache)
    nothing
end

function DiffEqBase.interp_summary(::Type{cacheType}, dense::Bool) where {cacheType <: TaylorMethodCache}
    dense ? "Taylor series polynomial evaluation" : "1st order linear"
end

if VERSION < v"1.9"
    # used when idxs gives back multiple values
    function OrdinaryDiffEq._ode_interpolant!(out, Θ, dt, y₀, y₁, k,
            cache::TaylorMethodCache, idxs, T::Type{Val{TI}}) where {TI}
        Θm1 = Θ - 1
        @inbounds for i in eachindex(out)
            out[i] = cache.uT[i](Θm1*dt)
        end
        out
    end
    # used when idxs gives back a single value
    function OrdinaryDiffEq._ode_interpolant(Θ, dt, y₀, y₁, k,
            cache::TaylorMethodCache, idxs, T::Type{Val{TI}}) where {TI}
        Θm1 = Θ - 1
        return cache.uT[idxs](Θm1*dt)
    end
else
    # used when idxs gives back multiple values
    function OrdinaryDiffEq._ode_interpolant!(out, Θ, dt, y₀, y₁, k,
            cache::TaylorMethodCache, idxs, T::Type{Val{TI}}, differential_vars) where {TI}
        Θm1 = Θ - 1
        @inbounds for i in eachindex(out)
            out[i] = cache.uT[i](Θm1*dt)
        end
        out
    end
    # used when idxs gives back a single value
    function OrdinaryDiffEq._ode_interpolant(Θ, dt, y₀, y₁, k,
            cache::TaylorMethodCache, idxs, T::Type{Val{TI}}, differential_vars) where {TI}
        Θm1 = Θ - 1
        return cache.uT[idxs](Θm1*dt)
    end
end

@inline TaylorIntegration.__jetcoeffs!(::Val{false}, f::ODEFunction, t, x::Taylor1{U}, params,
    rv::RetAlloc{Taylor1{U}}) where {U} = TaylorIntegration.__jetcoeffs!(Val(false), f.f, t, x, params, rv)
@inline TaylorIntegration.__jetcoeffs!(::Val{true},  f::ODEFunction, t, x::Taylor1{U}, params,
    rv::RetAlloc{Taylor1{U}}) where {U} = TaylorIntegration.__jetcoeffs!(Val(true), f.f, t, x, params, rv)
@inline TaylorIntegration.__jetcoeffs!(::Val{false}, f::ODEFunction, t, x::AbstractArray{Taylor1{U},N}, dx, xaux, params,
    rv::RetAlloc{Taylor1{U}}) where {U,N} = TaylorIntegration.__jetcoeffs!(Val(false), f.f, t, x, dx, xaux, params, rv)
@inline TaylorIntegration.__jetcoeffs!(::Val{true},  f::ODEFunction, t, x::AbstractArray{Taylor1{U},N}, dx, xaux, params,
    rv::RetAlloc{Taylor1{U}}) where {U,N} = TaylorIntegration.__jetcoeffs!(Val(true), f.f, t, x, dx, xaux, params, rv)

# NOTE: DynamicalODEFunction assumes x is a vector
# @inline TaylorIntegration.__jetcoeffs!(::Val{false}, f::DynamicalODEFunction, t, x::Taylor1{U}, params,
#     rv::RetAlloc{Taylor1{U}}) where {U} = TaylorIntegration.__jetcoeffs!(Val(false), f, t, x, params)
# @inline TaylorIntegration.__jetcoeffs!(::Val{true},  f::DynamicalODEFunction, t, x::Taylor1{U}, params,
#     rv::RetAlloc{Taylor1{U}}) where {U} = TaylorIntegration.__jetcoeffs!(Val(true), f, t, x, params, rv)
@inline TaylorIntegration.__jetcoeffs!(::Val{false}, f::DynamicalODEFunction, t, x::ArrayPartition, dx, xaux, params,
    rv::RetAlloc) = TaylorIntegration.jetcoeffs!(f, t, vec(x), vec(dx), xaux, params)
# NOTE: `parse_eqs=true` not implemented for `DynamicalODEFunction`
# @inline TaylorIntegration.__jetcoeffs!(::Val{true},  f::DynamicalODEFunction, t, x::ArrayPartition, dx, xaux, params,
#     rv::RetAlloc) = TaylorIntegration.__jetcoeffs!(Val(true), f, t, vec(x), vec(dx), params, rv)

TaylorIntegration._determine_parsing!(parse_eqs::Bool, f::ODEFunction, t, x, params) =
    TaylorIntegration._determine_parsing!(parse_eqs, f.f, t, x, params)
TaylorIntegration._determine_parsing!(parse_eqs::Bool, f::ODEFunction, t, x, dx, params) =
    TaylorIntegration._determine_parsing!(parse_eqs, f.f, t, x, dx, params)
end
