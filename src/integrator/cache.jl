# This file is part of the TaylorIntegration.jl package; MIT licensed

# AbstractTaylorIntegrationCache interface

abstract type AbstractTaylorIntegrationCache end

tv(c::AbstractTaylorIntegrationCache) = c.tv
xv(c::AbstractTaylorIntegrationCache) = c.xv
psol(c::AbstractTaylorIntegrationCache) = c.psol

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

function init_cache(cachetype::Type{ScalarCache}, dense::Val{D}, t0::T, x0::U, maxsteps::Int, order::Int) where {D, U, T}
    # Initialize the Taylor1 expansions
    t = t0 + Taylor1( T, order )
    x = Taylor1( x0, order )
    # Initialize cache
    return cachetype(
        Array{T}(undef, maxsteps + 1),
        Array{U}(undef, maxsteps + 1),
        init_psol(dense, maxsteps, 1, x),
        t,
        x)
end

function init_cache(cachetype::Type{ScalarCache}, ::Val{false}, trange::AbstractVector{T}, x0::U, maxsteps::Int, order::Int) where {U, T}
    # Initialize the Taylor1 expansions
    t0 = trange[1]
    t = t0 + Taylor1( T, order )
    x = Taylor1( x0, order )
    # Initialize cache
    nn = length(trange)
    cache = cachetype(
        trange,
        Array{U}(undef, nn),
        init_psol(Val(false), maxsteps, 1, x),
        t,
        x)
    fill!(cache.xv, T(NaN))
    return cache
end

# VectorCache

struct VectorCache{TV, XV, PSOL, XAUX} <: AbstractVectorCache
    tv::TV
    xv::XV
    psol::PSOL
    xaux::XAUX
end

function init_cache(cachetype::Type{VectorCache}, dense::Val{D}, t0::T, x::Vector{Taylor1{U}}, maxsteps::Int) where {D, U, T}
    dof = length(x)
    return cachetype(
        Array{T}(undef, maxsteps + 1),
        Array{U}(undef, dof, maxsteps + 1),
        init_psol(dense, maxsteps, dof, x),
        Array{Taylor1{U}}(undef, dof))
end

# VectorTRangeCache

struct VectorTRangeCache{TV, XV, PSOL, XAUX, X0, X1} <: AbstractVectorCache
    tv::TV
    xv::XV
    psol::PSOL
    xaux::XAUX
    x0::X0
    x1::X1
end

function init_cache(cachetype::Type{VectorTRangeCache}, ::Val{false}, trange::AbstractVector{T}, x::Vector{Taylor1{U}}, maxsteps::Int) where {U, T}
    nn = length(trange)
    dof = length(x)
    cache = cachetype(
        trange,
        Array{U}(undef, dof, nn),
        init_psol(Val(false), maxsteps, dof, x),
        Array{Taylor1{U}}(undef, dof),
        similar(constant_term.(x)),
        similar(constant_term.(x)))
    fill!(cache.x0, T(NaN))
    for ind in 1:nn
        @inbounds cache.xv[:,ind] .= cache.x0
    end
    return cache
end

# LyapunovSpectrumCache

struct LyapunovSpectrumCache{TV, XV, PSOL, XAUX, X0, Λ, ΛTSUM, ΔX, DΔX, JAC, VARSAUX, QQH, RRH, AJ, QI, VJ} <: AbstractVectorCache
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
end

function init_cache(cachetype::Type{LyapunovSpectrumCache}, dense, t0::T, x::Vector{Taylor1{U}}, maxsteps::Int) where {U, T}
    nx0 = length(x) # equals dof + dof^2
    dof = Int(sqrt(nx0 + 1/4) - 1/2)
    cache = cachetype(
        Array{T}(undef, maxsteps+1),
        Array{U}(undef, dof, maxsteps+1),
        nothing,
        Array{Taylor1{U}}(undef, nx0),
        getcoeff.(x, 0),
        Array{U}(undef, dof, maxsteps+1),
        similar(constant_term.(x[1:dof])),
        Array{TaylorN{Taylor1{U}}}(undef, dof),
        Array{TaylorN{Taylor1{U}}}(undef, dof),
        Array{Taylor1{U}}(undef, dof, dof),
        Array{Taylor1{U}}(undef, dof, dof, dof),
        Array{U}(undef, dof, dof),
        Array{U}(undef, dof, dof),
        Array{U}(undef, dof),
        Array{U}(undef, dof),
        Array{U}(undef, dof)
    )
    fill!(cache.jac, zero(x[1]))
    return cache
end

# LyapunovSpectrumTRangeCache

struct LyapunovSpectrumTRangeCache{TV, XV, PSOL, XAUX, X0, Q1, Λ, ΛTSUM, ΔX, DΔX, JAC, VARSAUX, QQH, RRH, AJ, QI, VJ} <: AbstractVectorCache
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
end

function init_cache(cachetype::Type{LyapunovSpectrumTRangeCache}, dense, trange::AbstractVector{T}, x::Vector{Taylor1{U}}, maxsteps::Int) where {U, T}
    nx0 = length(x) # equals dof + dof^2
    dof = Int(sqrt(nx0 + 1/4) - 1/2)
    nn = length(trange)
    cache = cachetype(
        trange,
        Array{U}(undef, dof, nn),
        nothing,
        Array{Taylor1{U}}(undef, nx0),
        similar(constant_term.(x)),
        similar(constant_term.(x[1:dof])),
        Array{U}(undef, dof, nn),
        similar(constant_term.(x[1:dof])),
        Array{TaylorN{Taylor1{U}}}(undef, dof),
        Array{TaylorN{Taylor1{U}}}(undef, dof),
        Array{Taylor1{U}}(undef, dof, dof),
        Array{Taylor1{U}}(undef, dof, dof, dof),
        Array{U}(undef, dof, dof),
        Array{U}(undef, dof, dof),
        Array{U}(undef, dof),
        Array{U}(undef, dof),
        Array{U}(undef, dof)
    )
    fill!(cache.xv, U(NaN))
    fill!(cache.λ, U(NaN))
    return cache
end
