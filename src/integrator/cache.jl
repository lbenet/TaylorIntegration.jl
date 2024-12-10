# This file is part of the TaylorIntegration.jl package; MIT licensed

# AbstractTaylorIntegrationCache interface

abstract type AbstractTaylorIntegrationCache end

tv(c::AbstractTaylorIntegrationCache) = c.tv
xv(c::AbstractTaylorIntegrationCache) = c.xv
psol(c::AbstractTaylorIntegrationCache) = c.psol

# AbstractTaylorIntegrationVectorCache interface

abstract type AbstractTaylorIntegrationVectorCache <: AbstractTaylorIntegrationCache end

xaux(c::AbstractTaylorIntegrationVectorCache) = c.xaux

# AbstractTaylorIntegrationCache

struct TaylorIntegrationScalarCache{TV, XV, PSOL} <: AbstractTaylorIntegrationCache
    tv::TV
    xv::XV
    psol::PSOL
end

# AbstractTaylorIntegrationVectorCache

struct TaylorIntegrationVectorCache{TV, XV, PSOL, XAUX} <: AbstractTaylorIntegrationVectorCache
    tv::TV
    xv::XV
    psol::PSOL
    xaux::XAUX
end

# TaylorIntegrationVectorTRangeCache

struct TaylorIntegrationVectorTRangeCache{TV, XV, PSOL, XAUX, X0, X1} <: AbstractTaylorIntegrationVectorCache
    tv::TV
    xv::XV
    psol::PSOL
    xaux::XAUX
    x0::X0
    x1::X1
end

# TaylorIntegrationLyapunovSpectrumCache

struct TaylorIntegrationLyapunovSpectrumCache{TV, XV, PSOL, XAUX, X0, Λ, ΛTSUM, ΔX, DΔX, JAC, VARSAUX, QQH, RRH, AJ, QI, VJ} <: AbstractTaylorIntegrationVectorCache
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

# TaylorIntegrationLyapunovSpectrumTRangeCache

struct TaylorIntegrationLyapunovSpectrumTRangeCache{TV, XV, PSOL, XAUX, X0, Q1, Λ, ΛTSUM, ΔX, DΔX, JAC, VARSAUX, QQH, RRH, AJ, QI, VJ} <: AbstractTaylorIntegrationVectorCache
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
