using TaylorIntegration
using StaticArrays
using AbstractTrees
using Test

@testset "Automatic Domain Splitting" begin
    @testset "2D Kepler problem" begin
        using TaylorIntegration: countnodes, timeshift!

        # This example is based upon section 3 of
        # https://doi.org/10.1007/s10569-015-9618-3.

        # Dynamical function
        @taylorize function kepler_eqs!(dq, q, params, t)
            dq[1] = q[3]
            dq[2] = q[4]
            rr = ( q[1]^2 + q[2]^2 )^(3/2)
            dq[3] = - q[1] / rr
            dq[4] = - q[2] / rr
        end
        # Initial conditons (plain)
        q00 = [1.0, 0.0, 0.0, sqrt(1.5)]
        # Jet transport variables
        dq = set_variables("dx dy", numvars = 2, order = 5)
        # Initial conditions (jet transport)
        q0 = q00 .+ [0.008, 0.08, 0.0, 0.0] .* vcat(dq, dq)
        # Initial time
        t0 = 0.0
        # Final time
        tmax = 34.85
        # Taylor1 order
        order = 25
        # Splitting tolerance
        stol = 0.008
        # Absolute tolerance
        abstol = 1e-20
        # Dynamical function parameters
        params = nothing
        # Maximum allowed splits
        maxsplits = 25
        # Maximum allowed steps
        maxsteps = 1_000

        # ADS taylorinteg (parse_eqs = false)
        # Warmup
        q1 = ADSTaylorSolution([-1.0, -1.0], [1.0, 1.0], q0)
        taylorinteg(kepler_eqs!, q1, t0, tmax, order, stol, abstol, params;
                    maxsplits = 1, maxsteps = 1, parse_eqs = false);
        # Full integration
        q1 = ADSTaylorSolution([-1.0, -1.0], [1.0, 1.0], q0)
        taylorinteg(kepler_eqs!, q1, t0, tmax, order, stol, abstol, params;
                    maxsplits = maxsplits, maxsteps = maxsteps, parse_eqs = false);

        @test isa(q1, ADSTaylorSolution{Float64, 2, 4})
        @test iszero(q1.depth)
        @test q1.t == t0
        @test q1.lo == [-1.0, -1.0]
        @test q1.hi == [1.0, 1.0]
        @test q1.x == q0
        @test constant_term.(q1.p) == q0
        @test isnothing(q1.parent)
        @test isa(q1.left, ADSTaylorSolution{Float64, 2, 4})
        @test isnothing(q1.right)
        l1 = collect(Leaves(q1))
        @test all(getfield.(l1, Ref(:t)) .== tmax)

        @test isone(countnodes(q1, 0))
        d1 = maximum(getfield.(l1, Ref(:depth)))
        @test countnodes(q1, d1) == 2
        @test iszero(countnodes(q1, d1 + 1))

        @test iszero(countnodes(q1, prevfloat(t0)))
        @test isone(countnodes(q1, t0))
        @test isone(countnodes(q1, t0 + 1.0))
        @test countnodes(q1, tmax - 1.0) == maxsplits
        @test countnodes(q1, tmax) == maxsplits
        @test iszero(countnodes(q1, nextfloat(tmax)))

        # ADS taylorinteg (parse_eqs = true)
        # Warmup
        q2 = ADSTaylorSolution([-1.0, -1.0], [1.0, 1.0], q0)
        taylorinteg(kepler_eqs!, q2, t0, tmax, order, stol, abstol, params;
                    maxsplits = 1, maxsteps = 1, parse_eqs = true);
        # Full integration
        q2 = ADSTaylorSolution([-1.0, -1.0], [1.0, 1.0], q0)
        taylorinteg(kepler_eqs!, q2, t0, tmax, order, stol, abstol, params;
                    maxsplits = maxsplits, maxsteps = maxsteps, parse_eqs = true);

        @test isa(q2, ADSTaylorSolution{Float64, 2, 4})
        @test iszero(q2.depth)
        @test q2.t == t0
        @test q2.lo == [-1.0, -1.0]
        @test q2.hi == [1.0, 1.0]
        @test q2.x == q0
        @test constant_term.(q2.p) == q0
        @test isnothing(q2.parent)
        @test isa(q2.left, ADSTaylorSolution{Float64, 2, 4})
        @test isnothing(q2.right)
        l2= collect(Leaves(q2))
        @test all(getfield.(l2, Ref(:t)) .== tmax)

        @test isone(countnodes(q2, 0))
        d2 = maximum(getfield.(l2, Ref(:depth)))
        @test countnodes(q2, d2) == 2
        @test iszero(countnodes(q2, d2 + 1))

        @test iszero(countnodes(q2, prevfloat(t0)))
        @test isone(countnodes(q2, t0))
        @test isone(countnodes(q2, t0 + 1.0))
        @test countnodes(q2, tmax - 1.0) == maxsplits
        @test countnodes(q2, tmax) == maxsplits
        @test iszero(countnodes(q2, nextfloat(tmax)))

        # Compatibility between parse_eqs = false/true

        for (n1, n2) in zip(PreOrderDFS(q1), PreOrderDFS(q2))
            @test n1.depth == n2.depth
            @test n1.t == n2.t
            @test n1.lo == n2.lo
            @test n1.hi == n2.hi
            @test n1.x == n2.x
            @test n1.p == n2.p
        end

        lo1, hi1, x1 = q1(t0)
        lo2, hi2, x2 = q2(t0)
        @test size(lo1) == size(hi1) == size(lo2) == size(hi2) == (2, 1)
        @test all(lo1 .== lo2 .== q1.lo .== q2.lo)
        @test all(hi1 .== hi2 .== q1.hi .== q2.hi)
        @test size(x1) == size(x2) == (4, 1)
        @test all(x1 .== x2 .== q1.x .== q2.x)

        lo1, hi1, x1 = q1(tmax)
        lo2, hi2, x2 = q2(tmax)
        @test size(lo1) == size(hi1) == size(lo2) == size(hi2) == (2, maxsplits)
        @test lo1 == lo2
        @test hi1 == hi2
        @test size(x1) == size(x2) == (4, maxsplits)
        @test x1 == x2

        y1 = q1(t0, [0.1, 0.1])
        y2 = q2(t0, [0.1, 0.1])
        @test y1 == y2
        y1 = q1(tmax, [0.1, 0.1])
        y2 = q2(tmax, [0.1, 0.1])
        @test y1 == y2

        # ADS vs Monte Carlo both in cartesian coordinates and keplerian elements

        # Semimajor axis and eccentricity
        function ae(rv)
            x, y, u, v = rv
            r = sqrt(x^2 + y^2)
            vsq = u^2 + v^2
            a = 1 / ( (2/r)- vsq )
            hsq = (x*v - y*u)^2
            e = sqrt(1 - hsq/a)
            return a, e
        end
        # 16 points in the boundary of the domain
        side = LinRange(-1, 1, 5)
        boundary = vcat(
            map(x -> [x, -1], side), # Bottom
            map(x -> [1, x], side),  # Right
            map(x -> [-x, 1], side),  # Top
            map(x -> [-1, -x], side)  # Left
        )
        unique!(boundary)

        for s in boundary
            sol = taylorinteg(kepler_eqs!, q0(s), t0, tmax, order, abstol, params;
                              maxsteps, parse_eqs = false, dense = false)
            rv0mc = sol.x[1, :]
            rvfmc = sol.x[end, :]

            rv0ads, rvfads = q1(t0, s), q1(tmax, s)

            @test rv0ads == rv0mc
            @test maximum(@. abs((rvfads - rvfmc) / rvfmc)) < 0.006

            a0, e0 = ae(rv0ads)
            af, ef = ae(rvfads)

            @test abs((af - a0) / a0) < 0.015
            @test abs((ef - e0) / e0) < 0.009

            sol = taylorinteg(kepler_eqs!, q0(s), t0, tmax, order, abstol, params;
                              maxsteps, parse_eqs = true, dense = false)
            rv0mc = sol.x[1, :]
            rvfmc = sol.x[end, :]

            rv0ads, rvfads = q2(t0, s), q2(tmax, s)

            @test rv0ads == rv0mc
            @test maximum(@. abs((rvfads - rvfmc) / rvfmc)) < 0.006

            a0, e0 = ae(rv0ads)
            af, ef = ae(rvfads)

            @test abs((af - a0) / a0) < 0.015
            @test abs((ef - e0) / e0) < 0.009
        end

        # timeshift!

        timeshift!(q2, 1.0)
        for (n1, n2) in zip(PreOrderDFS(q1), PreOrderDFS(q2))
            @test n2.t == n1.t + 1.0
        end
    end
end