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

# AbstractTaylorIntegrationVectorCache

struct TaylorIntegrationVectorTRangeCache{TV, XV, PSOL, XAUX, X0, X1} <: AbstractTaylorIntegrationVectorCache
    tv::TV
    xv::XV
    psol::PSOL
    xaux::XAUX
    x0::X0
    x1::X1
end
x0(c::TaylorIntegrationVectorTRangeCache) = c.x0
x1(c::TaylorIntegrationVectorTRangeCache) = c.x1
