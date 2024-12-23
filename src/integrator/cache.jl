# This file is part of the TaylorIntegration.jl package; MIT licensed

# AbstractTaylorIntegrationCache interface

abstract type AbstractTaylorIntegrationCache end

tv(c::AbstractTaylorIntegrationCache) = c.tv
xv(c::AbstractTaylorIntegrationCache) = c.xv
psol(c::AbstractTaylorIntegrationCache) = c.psol
t(c::AbstractTaylorIntegrationCache) = c.t
x(c::AbstractTaylorIntegrationCache) = c.x

# AbstractVectorCache interface

abstract type AbstractVectorCache <: AbstractTaylorIntegrationCache end

xaux(c::AbstractVectorCache) = c.xaux

# ScalarCache

struct ScalarCache{TV, XV, PSOL, T, X} <: AbstractTaylorIntegrationCache
    tv::TV
    xv::XV
    psol::PSOL
    t::T
    x::X
end

# VectorCache

struct VectorCache{TV, XV, PSOL, XAUX, T, X, DX} <: AbstractVectorCache
    tv::TV
    xv::XV
    psol::PSOL
    xaux::XAUX
    t::T
    x::X
    dx::DX
end

# VectorTRangeCache

struct VectorTRangeCache{TV, XV, PSOL, XAUX, X0, X1, T, X, DX} <: AbstractVectorCache
    tv::TV
    xv::XV
    psol::PSOL
    xaux::XAUX
    x0::X0
    x1::X1
    t::T
    x::X
    dx::DX
end

# LyapunovSpectrumCache

struct LyapunovSpectrumCache{TV, XV, PSOL, XAUX, X0, Λ, ΛTSUM, ΔX, DΔX, JAC, VARSAUX, QQH, RRH, AJ, QI, VJ, T, X, DX, JT, DVARS} <: AbstractVectorCache
    tv::TV
    xv::XV
    psol::PSOL
    xaux::XAUX
    x0::X0
    λ::Λ
    λtsum::ΛTSUM
    δx::ΔX
    dδx::DΔX
    jac::JAC
    varsaux::VARSAUX
    QH::QQH
    RH::RRH
    aⱼ::AJ
    qᵢ::QI
    vⱼ::VJ
    t::T
    x::X
    dx::DX
    jt::JT
    dvars::DVARS
end

# LyapunovSpectrumTRangeCache

struct LyapunovSpectrumTRangeCache{TV, XV, PSOL, XAUX, X0, Q1, Λ, ΛTSUM, ΔX, DΔX, JAC, VARSAUX, QQH, RRH, AJ, QI, VJ, T, X, DX, JT, DVARS} <: AbstractVectorCache
    tv::TV
    xv::XV
    psol::PSOL
    xaux::XAUX
    x0::X0
    q1::Q1
    λ::Λ
    λtsum::ΛTSUM
    δx::ΔX
    dδx::DΔX
    jac::JAC
    varsaux::VARSAUX
    QH::QQH
    RH::RRH
    aⱼ::AJ
    qᵢ::QI
    vⱼ::VJ
    t::T
    x::X
    dx::DX
    jt::JT
    dvars::DVARS
end

# init_expansions

function init_expansions(t0::T, x0::U, order::Int) where {T, U}
    t = t0 + Taylor1( T, order )
    x = Taylor1( x0, order )
    return t, x
end

function init_expansions(t0::T, q0::Vector{U}, order::Int) where {T, U}
    dof = length(q0)
    t = t0 + Taylor1( T, order )
    x = Array{Taylor1{U}}(undef, dof)
    dx = Array{Taylor1{U}}(undef, dof)
    x .= Taylor1.( q0, order )
    dx .= Taylor1.( zero.(q0), order )
    return t, x, dx
end

# init_cache

function init_cache(dense::Val{D}, t0::T, x0::U, maxsteps::Int, order::Int) where {D, U, T}
    # Initialize the Taylor1 expansions
    t, x = init_expansions(t0, x0, order)
    # Initialize cache
    return ScalarCache(
        Array{T}(undef, maxsteps + 1),
        Array{U}(undef, maxsteps + 1),
        init_psol(dense, maxsteps, 1, x),
        t,
        x)
end

function init_cache(::Val{false}, trange::AbstractVector{T}, x0::U, maxsteps::Int, order::Int) where {U, T}
    # Initialize the Taylor1 expansions
    t0 = trange[1]
    t, x = init_expansions(t0, x0, order)
    # Initialize cache
    nn = length(trange)
    cache = ScalarCache(
        trange,
        Array{U}(undef, nn),
        init_psol(Val(false), maxsteps, 1, x),
        t,
        x)
    fill!(cache.xv, T(NaN))
    return cache
end

function init_cache(dense::Val{D}, t0::T, q0::Vector{U}, maxsteps::Int, order::Int) where {D, U, T}
    # Initialize the vector of Taylor1 expansions
    t, x, dx = init_expansions(t0, q0, order)
    # Initialize cache
    dof = length(q0)
    return VectorCache(
        Array{T}(undef, maxsteps + 1),
        Array{U}(undef, dof, maxsteps + 1),
        init_psol(dense, maxsteps, dof, x),
        Array{Taylor1{U}}(undef, dof),
        t,
        x,
        dx)
end

function init_cache(::Val{false}, trange::AbstractVector{T}, q0::Vector{U}, maxsteps::Int, order::Int) where {U, T}
    # Initialize the vector of Taylor1 expansions
    t0 = trange[1]
    t, x, dx = init_expansions(t0, q0, order)
    # Initialize cache
    nn = length(trange)
    dof = length(q0)
    cache = VectorTRangeCache(
        trange,
        Array{U}(undef, dof, nn),
        init_psol(Val(false), maxsteps, dof, x),
        Array{Taylor1{U}}(undef, dof),
        similar(q0),
        similar(q0),
        t,
        x,
        dx)
    fill!(cache.x0, T(NaN))
    for ind in 1:nn
        @inbounds cache.xv[:,ind] .= cache.x0
    end
    return cache
end

function init_cache_lyap(t0::T, q0::Vector{U}, maxsteps::Int, order::Int) where {U, T}
    # Initialize the vector of Taylor1 expansions
    dof = length(q0)
    jt = Matrix{U}(I, dof, dof)
    x0 = vcat(q0, reshape(jt, dof*dof))
    t, x, dx = init_expansions(t0, x0, order)
    # Initialize cache
    nx0 = length(x0)
    dvars = Array{TaylorN{Taylor1{U}}}(undef, dof)
    cache = LyapunovSpectrumCache(
        Array{T}(undef, maxsteps+1),
        Array{U}(undef, dof, maxsteps+1),
        nothing,
        Array{Taylor1{U}}(undef, nx0),
        x0,
        Array{U}(undef, dof, maxsteps+1),
        similar(q0),
        Array{TaylorN{Taylor1{U}}}(undef, dof),
        Array{TaylorN{Taylor1{U}}}(undef, dof),
        Array{Taylor1{U}}(undef, dof, dof),
        Array{Taylor1{U}}(undef, dof, dof, dof),
        Array{U}(undef, dof, dof),
        Array{U}(undef, dof, dof),
        Array{U}(undef, dof),
        Array{U}(undef, dof),
        Array{U}(undef, dof),
        t,
        x,
        dx,
        jt,
        dvars
    )
    fill!(cache.jac, zero(x[1]))
    return cache
end

function init_cache_lyap(trange::AbstractVector{T}, q0::Vector{U}, maxsteps::Int, order::Int) where {U, T}
    # Initialize the vector of Taylor1 expansions
    t0 = trange[1]
    dof = length(q0)
    jt = Matrix{U}(I, dof, dof)
    x0 = vcat(q0, reshape(jt, dof*dof))
    t, x, dx = init_expansions(t0, x0, order)
    # Initialize cache
    nn = length(trange)
    nx0 = length(x0)
    dvars = Array{TaylorN{Taylor1{U}}}(undef, dof)
    cache = LyapunovSpectrumTRangeCache(
        trange,
        Array{U}(undef, dof, nn),
        nothing,
        Array{Taylor1{U}}(undef, nx0),
        similar(x0),
        similar(q0),
        Array{U}(undef, dof, nn),
        similar(q0),
        Array{TaylorN{Taylor1{U}}}(undef, dof),
        Array{TaylorN{Taylor1{U}}}(undef, dof),
        Array{Taylor1{U}}(undef, dof, dof),
        Array{Taylor1{U}}(undef, dof, dof, dof),
        Array{U}(undef, dof, dof),
        Array{U}(undef, dof, dof),
        Array{U}(undef, dof),
        Array{U}(undef, dof),
        Array{U}(undef, dof),
        t,
        x,
        dx,
        jt,
        dvars
    )
    fill!(cache.xv, U(NaN))
    fill!(cache.λ, U(NaN))
    return cache
end

 # update!

function update!(cache::ScalarCache, t0::T, x0::U) where {T, U}
    @unpack t, x = cache
    @inbounds x[0] = x0
    @inbounds t[0] = t0
    return nothing
end

function update!(cache::AbstractVectorCache, t0::T, x0::Vector{U}) where {T, U}
    @unpack t, x = cache
    @inbounds for i in eachindex(x0)
        x[i][0] = x0[i]
    end
    @inbounds t[0] = t0
    return nothing
end