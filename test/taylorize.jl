# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorIntegration, OrdinaryDiffEq
using Test
using LinearAlgebra: norm
using InteractiveUtils: methodswith
using Elliptic
using Base.Threads
import Pkg
using Logging
import Logging: Warn

@testset "Testing `taylorize.jl`" begin

    # Constants for the integrations
    local TI = TaylorIntegration
    local _order = 20
    local _abstol = 1.0e-20
    local t0 = 0.0
    local tf = 1000.0

    unable_to_parse(f) = """Unable to use the parsed method of `jetcoeffs!` for `$f`,
    despite of having `parse_eqs=true`, due to some internal error.
    Using `parse_eqs = false`."""

    max_iters_reached() = "Maximum number of integration steps reached; exiting.\n"

    # Scalar integration
    @testset "Scalar case: xdot(x, p, t) = b-x^2" begin
        # Use a global constant
        b1 = 3.0
        @taylorize xdot1(x, p, t) = b1-x^2
        @test (@isdefined xdot1)

        x0 = 1.0
        sol1 = taylorinteg( xdot1, x0, t0, tf, _order, _abstol, maxsteps=1000,
            dense=false, parse_eqs=false)
        tv1 = sol1.t
        xv1 = sol1.x
        sol1p = (@test_logs min_level=Logging.Warn taylorinteg(
            xdot1, x0, t0, tf, _order, _abstol, maxsteps=1000, dense=false))
        tv1p = sol1p.t
        xv1p = sol1p.x

        @test length(tv1) == length(tv1p)
        @test iszero( norm(tv1-tv1p, Inf) )
        @test iszero( norm(xv1-xv1p, Inf) )

        # Use `local` constants
        @taylorize xdot2(x, p, t) = (local b2 = 3; b2-x^2)
        @test (@isdefined xdot2)

        sol2 = taylorinteg( xdot2, x0, t0, tf, _order, _abstol, maxsteps=1000,
            dense=false, parse_eqs=false)
        tv2 = sol2.t
        xv2 = sol2.x
        sol2p = (@test_logs min_level=Logging.Warn taylorinteg(
            xdot2, x0, t0, tf, _order, _abstol, maxsteps=1000, dense=false))
        tv2p = sol2p.t
        xv2p = sol2p.x

        @test length(tv2) == length(tv2p)
        @test iszero( norm(tv2-tv2p, Inf) )
        @test iszero( norm(xv2-xv2p, Inf) )

        # Passing a parameter
        @taylorize xdot3(x, p, t) = p-x^2
        @test (@isdefined xdot3)

        sol3 = taylorinteg( xdot3, x0, t0, tf, _order, _abstol, b1, maxsteps=1000,
            dense=false, parse_eqs=false)
        tv3 = sol3.t
        xv3 = sol3.x
        sol3p = (@test_logs min_level=Logging.Warn taylorinteg(
            xdot3, x0, t0, tf, _order, _abstol, b1, maxsteps=1000, dense=false))
        tv3p = sol3p.t
        xv3p = sol3p.x

        @test length(tv3) == length(tv3p)
        @test iszero( norm(tv3-tv3p, Inf) )
        @test iszero( norm(xv3-xv3p, Inf) )

        # Comparing integrations
        @test length(tv1p) == length(tv2p) == length(tv3p)
        @test iszero( norm(tv1p-tv2p, Inf) )
        @test iszero( norm(tv1p-tv3p, Inf) )
        @test iszero( norm(xv1p-xv2p, Inf) )
        @test iszero( norm(xv1p-xv3p, Inf) )

        sol4 = taylorinteg( xdot2, x0, t0:0.5:tf, _order, _abstol, maxsteps=1000,
            parse_eqs=false)
        xv4 = sol4.x
        sol4p = (@test_logs min_level=Logging.Warn taylorinteg(
            xdot2, x0, t0:0.5:tf, _order, _abstol, maxsteps=1000))
        xv4p = sol4p.x
        @test iszero( norm(xv4-xv4p, Inf) )

        # Compare to exact solution
        exact_sol(t, b, x0) = sqrt(b)*((sqrt(b)+x0)-(sqrt(b)-x0)*exp(-2sqrt(b)*t)) /
            ((sqrt(b)+x0)+(sqrt(b)-x0)*exp(-2sqrt(b)*t))
        @test norm(xv1p[end] - exact_sol(tv1p[end], b1, x0), Inf) < 1.0e-15

        # Check that the parsed `jetcoeffs` produces the correct series in `x` and no error
        # TODO: Use metaprogramming here
        tT = t0 + Taylor1(_order)
        xT = x0 + zero(tT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(xdot1), tT, xT, nothing))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(xdot1), tT, xT, nothing, rv)
        @test xT ≈ exact_sol(tT, b1, x0)

        xT = x0 + zero(tT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(xdot2), tT, xT, nothing))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(xdot2), tT, xT, nothing, rv)
        @test xT ≈ exact_sol(tT, 3.0, x0)

        xT = x0 + zero(tT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(xdot2), tT, xT, b1))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(xdot2), tT, xT, b1, rv)
        @test xT ≈ exact_sol(tT, b1, x0)

        # The macro returns a (parsed) jetcoeffs! function which yields a `MethodError`
        # (TaylorSeries.identity! requires *two* Taylor series and an integer) and
        # therefore it runs with the default jetcoeffs! method (i.e., parse_eqs=false).
        # A warning is issued.
        @taylorize function xdot2_err(x, p, t)
            b2 = 3  # This line is parsed as `identity!` and yields a MethodError;
                    # correct this by replacing it to `b2 = 3 + zero(t)`
            return b2-x^2
        end

        @test (@isdefined xdot2_err)
        @test !isempty(methodswith(Val{xdot2_err}, TI.jetcoeffs!))
        sol2e = (@test_logs (Warn, unable_to_parse(xdot2_err)) taylorinteg(
                    xdot2_err, x0, t0, tf, _order, _abstol, maxsteps=1000, dense=false))
        tv2e = sol2e.t
        xv2e = sol2e.x
        @test length(tv2) == length(tv2e)
        @test iszero( norm(tv2-tv2e, Inf) )
        @test iszero( norm(xv2-xv2e, Inf) )
        xT = x0 + zero(tT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(xdot2_err), tT, xT, nothing))
        @test_throws MethodError TI.jetcoeffs!(
            Val(xdot2_err), tT, xT, nothing, rv)

        # Output includes Taylor polynomial solution
        sol3 = (@test_logs (Warn, max_iters_reached()) taylorinteg(
            xdot3, x0, t0, tf, _order, _abstol, 0.0, maxsteps=2))
        tv3t = sol3.t
        xv3t = sol3.x
        psol3t = sol3.p
        @test length(psol3t) == 2
        @test xv3t[1] == x0
        @test psol3t[1] == Taylor1([(-1.0)^i for i=0:_order])
        @test xv3t[2] == evaluate(psol3t[1], tv3t[2]-tv3t[1])
        @test xv3t[2] == sol3(tv3t[2])
    end


    @testset "Scalar case: xdot(x, p, t) = -10" begin
        # Return an expression
        xdot1(x, p, t) = -10 + zero(t)         # `zero(t)` is needed; cf #20
        @taylorize xdot1_parsed(x, p, t) = -10 # `zero(t)` can be avoided here !

        @test (@isdefined xdot1_parsed)
        sol1  = taylorinteg( xdot1, 10, 1, 20.0, _order, _abstol, dense=false,
            parse_eqs=false)
        tv1, xv1 = sol1.t, sol1.x
        sol1p = (@test_logs min_level=Logging.Warn taylorinteg(
            xdot1_parsed, 10.0, 1.0, 20.0, _order, _abstol, dense=false))
        tv1p, xv1p = sol1p.t, sol1p.x

        @test length(tv1) == length(tv1p)
        @test iszero( norm(tv1-tv1p, Inf) )
        @test iszero( norm(xv1-xv1p, Inf) )

        # Use `local` expression
        @taylorize function xdot2(x, p, t)
            local ggrav = -10 + zero(t)  # zero(t) is needed for the `parse_eqs=false` case
            tmp = ggrav  # needed to avoid an error when parsing
        end

        @test (@isdefined xdot2)

        sol2  = taylorinteg( xdot2, 10.0, 1.0, 20.0, _order, _abstol, dense=false,
            parse_eqs=false)
        tv2, xv2 = sol2.t, sol2.x
        sol2p = (@test_logs min_level=Logging.Warn taylorinteg(
            xdot2, 10.0, 1, 20.0, _order, _abstol, dense=false))
        tv2p, xv2p = sol2p.t, sol2p.x

        @test length(tv2) == length(tv2p)
        @test iszero( norm(tv2-tv2p, Inf) )
        @test iszero( norm(xv2-xv2p, Inf) )

        # Passing a parameter
        @taylorize xdot3(x, p, t) = p + zero(t) # `zero(t)` can be avoided here

        @test (@isdefined xdot3)

        sol3  = taylorinteg( xdot3, 10.0, 1.0, 20.0, _order, _abstol, -10,
            dense=false, parse_eqs=false)
        tv3, xv3 = sol3.t, sol3.x
        sol3p = (@test_logs min_level=Logging.Warn taylorinteg(
            xdot3, 10, 1.0, 20.0, _order, _abstol, -10, dense=false))
        tv3p, xv3p = sol3p.t, sol3p.x

        @test length(tv3) == length(tv3p)
        @test iszero( norm(tv3-tv3p, Inf) )
        @test iszero( norm(xv3-xv3p, Inf) )

        # Comparing integrations
        @test length(tv1p) == length(tv2p) == length(tv3p)
        @test iszero( norm(tv1p-tv2p, Inf) )
        @test iszero( norm(tv1p-tv3p, Inf) )
        @test iszero( norm(xv1p-xv2p, Inf) )
        @test iszero( norm(xv1p-xv3p, Inf) )

        # Compare to exact solution
        exact_sol(t, g, x0) = x0 + g*(t-1.0)
        @test norm(xv1p[end] - exact_sol(20, -10, 10), Inf) < 10eps(exact_sol(20, -10, 10))

        # Check that the parsed `jetcoeffs` produces the correct series in `x` and no errors
        tT = 1.0 + Taylor1(_order)
        xT = 10.0 + zero(tT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(xdot1_parsed), tT, xT, nothing))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(xdot1_parsed), tT, xT, nothing, rv)
        @test xT ≈ exact_sol(tT, -10, 10)

        xT = 10.0 + zero(tT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(xdot2), tT, xT, nothing))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(xdot2), tT, xT, nothing, rv)
        @test xT ≈ exact_sol(tT, -10, 10)

        xT = 10.0 + zero(tT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(xdot3), tT, xT, -10))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(xdot3), tT, xT, -10, rv)
        @test xT ≈ exact_sol(tT, -10, 10)
    end


    # Pendulum integration
    @testset "Integration of the pendulum and DiffEqs interface" begin
        @taylorize function pendulum!(dx::Array{T,1}, x::Array{T,1}, p, t) where {T}
            dx[1] = x[2]
            dx[2] = -sin( x[1] )
            nothing
        end

        @test (@isdefined pendulum!)
        q0 = [pi-0.001, 0.0]
        sol2 = taylorinteg(pendulum!, q0, t0, tf, _order, _abstol, parse_eqs=false,
            maxsteps=5000, dense=false)
        tv2, xv2 = sol2.t, sol2.x

        sol2p = (@test_logs min_level=Logging.Warn taylorinteg(
            pendulum!, q0, t0, tf, _order, _abstol, maxsteps=5000, dense=false))
        tv2p, xv2p = sol2p.t, sol2p.x

        @test length(tv2) == length(tv2p)
        @test iszero( norm(tv2-tv2p, Inf) )
        @test iszero( norm(xv2-xv2p, Inf) )

        prob = ODEProblem(pendulum!, q0, (t0, tf), nothing) # no parameters
        sol1 = solve(prob, TaylorMethod(_order), abstol=_abstol, parse_eqs=true)
        sol2 = solve(prob, TaylorMethod(_order), abstol=_abstol, parse_eqs=false)
        sol3 = solve(prob, TaylorMethod(_order), abstol=_abstol)

        @test sol1.t == sol2.t == sol3.t == tv2p
        @test sol1.u[end] == sol2.u[end] == sol3.u[end] == xv2p[end,1:2]

        # Check that the parsed `jetcoeffs!` produces the correct series in `x` and no errors
        tT = t0 + Taylor1(_order)
        qT = q0 .+ zero(tT)
        dqT = similar(qT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(pendulum!), tT, qT, dqT, nothing))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(pendulum!), tT, qT, dqT, nothing, rv)
    end


    # Complex dependent variables
    @testset "Complex dependent variable" begin
        cc = complex(0.0,1.0)
        @taylorize eqscmplx(x, p, t) = cc*x

        @test (@isdefined eqscmplx)
        cx0 = complex(1.0, 0.0)
        sol1 = taylorinteg(eqscmplx, cx0, t0, tf, _order, _abstol, maxsteps=1500,
            parse_eqs=false, dense=false)
        tv1, xv1 = sol1.t, sol1.x
        sol1p = (@test_logs min_level=Logging.Warn taylorinteg(
            eqscmplx, cx0, t0, tf, _order, _abstol, maxsteps=1500, dense=false))
        tv1p, xv1p = sol1p.t, sol1p.x

        @test length(tv1) == length(tv1p)
        @test iszero( norm(tv1-tv1p, Inf) )
        @test iszero( norm(xv1-xv1p, Inf) )

        # Using `local` for the constant value
        @taylorize eqscmplx2(x, p, t) = (local cc1 = Complex(0.0,1.0); cc1*x)

        @test (@isdefined eqscmplx2)
        sol2 = taylorinteg(eqscmplx2, cx0, t0, tf, _order, _abstol, maxsteps=1500,
            parse_eqs=false, dense=false)
        tv2, xv2 = sol2.t, sol2.x
        sol2p = (@test_logs min_level=Logging.Warn taylorinteg(
            eqscmplx2, cx0, t0, tf, _order, _abstol, maxsteps=1500, dense=false))
        tv2p, xv2p = sol2p.t, sol2p.x

        @test length(tv2) == length(tv2p)
        @test iszero( norm(tv2-tv2p, Inf) )
        @test iszero( norm(xv2-xv2p, Inf) )

        # Passing a parameter
        @taylorize eqscmplx3(x, p, t) = p*x
        @test (@isdefined eqscmplx3)
        sol3 = taylorinteg(eqscmplx3, cx0, t0, tf, _order, _abstol, cc, maxsteps=1500,
            parse_eqs=false, dense=false)
        tv3, xv3 = sol3.t, sol3.x
        sol3p = (@test_logs min_level=Logging.Warn taylorinteg(
            eqscmplx3, cx0, t0, tf, _order, _abstol, cc, maxsteps=1500, dense=false))
        tv3p, xv3p = sol3p.t, sol3p.x

        @test length(tv3) == length(tv3p)
        @test iszero( norm(tv3-tv3p, Inf) )
        @test iszero( norm(xv3-xv3p, Inf) )

        # Comparing integrations
        @test length(tv1p) == length(tv2p) == length(tv3p)
        @test iszero( norm(tv1p-tv2p, Inf) )
        @test iszero( norm(tv1p-tv3p, Inf) )
        @test iszero( norm(xv1p-xv2p, Inf) )
        @test iszero( norm(xv1p-xv3p, Inf) )

        # Compare to exact solution
        exact_sol(t, p, x0) = x0*exp(p*t)
        nn = norm.(abs2.(xv1p)-abs2.(exact_sol.(tv1p, cc, cx0)), Inf)
        @test maximum(nn) < 1.0e-14

        eqs1(z, p, t) = -z
        eqs2(z, p, t) = im*z
        @taylorize function eqs3!(Dz, z, p, t)
            local _eltype_z_ = eltype(z)
            tmp = Array{_eltype_z_}(undef, 2)
            tmp[1] = eqs1(z[1], p, t)
            tmp[2] = eqs2(z[2], p, t)
            Dz[1] = tmp[1]
            Dz[2] = tmp[2]
            nothing
        end
        @test (@isdefined eqs3!)
        z0 = complex(0.0, 1.0)
        zz0 = [z0, z0]
        ts = 0.0:pi:2pi

        zsol = taylorinteg(eqs3!, zz0, ts, _order, _abstol, parse_eqs=false, maxsteps=10)
        tz, xz = zsol.t, zsol.x
        zsolp = (@test_logs min_level=Logging.Warn taylorinteg(
            eqs3!, zz0, ts, _order, _abstol, maxsteps=10))
        tzp, xzp = zsolp.t, zsolp.x
        @test length(tz) == length(tzp)
        @test iszero( norm(tz-tzp, Inf) )
        @test iszero( norm(xz-xzp, Inf) )

        tT = t0 + Taylor1(_order)
        zT = zz0 .+ zero(tT)
        dzT = similar(zT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(eqs3!), tT, zT, dzT, nothing))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(eqs3!), tT, zT, dzT, nothing, rv)
        @test typeof(rv.v0) == Vector{Taylor1{ComplexF64}}
        @test typeof(rv.v1) == Vector{Vector{Taylor1{ComplexF64}}}
        @test zT[1] == exact_sol(tT, -1, z0)
        @test zT[2] == exact_sol(tT, z0, z0)
    end


    @testset "Time-dependent integration (with and without `local` vars)" begin
        @taylorize function integ_cos1(x, p, t)
            y = cos(t)
            return y
        end
        @taylorize function integ_cos2(x, p, t)
            local y = cos(t)  # allows to calculate directly `cos(t)` *once*
            yy = y            # needed to avoid an error
            return yy
        end

        @test (@isdefined integ_cos1)
        @test (@isdefined integ_cos2)

        sol11 = taylorinteg(integ_cos1, 0.0, 0.0, pi, _order, _abstol, parse_eqs=false,
            dense=false)
        tv11, xv11 = sol11.t, sol11.x
        sol12 = (@test_logs min_level=Logging.Warn taylorinteg(
            integ_cos1, 0.0, 0.0, pi, _order, _abstol, dense=false))
        tv12, xv12 = sol12.t, sol12.x
        @test length(tv11) == length(tv12)
        @test iszero( norm(tv11-tv12, Inf) )
        @test iszero( norm(xv11-xv12, Inf) )

        sol21 = taylorinteg(integ_cos2, 0.0, 0.0, pi, _order, _abstol, parse_eqs=false,
            dense=false)
        tv21, xv21 = sol21.t, sol21.x
        sol22 = (@test_logs min_level=Logging.Warn taylorinteg(
            integ_cos2, 0.0, 0.0, pi, _order, _abstol, dense=false))
        tv22, xv22 = sol22.t, sol22.x
        @test length(tv21) == length(tv22)
        @test iszero( norm(tv21-tv22, Inf) )
        @test iszero( norm(xv21-xv22, Inf) )

        @test iszero( norm(tv12-tv22, Inf) )
        @test iszero( norm(xv12-xv22, Inf) )

        # Compare to exact solution
        @test norm(xv11[end] - sin(tv11[end]), Inf) < 1.0e-15
    end


    @testset "Time-dependent integration (vectorial case)" begin
        @taylorize function integ_vec(dx, x, p, t)
            dx[1] = cos(t)
            dx[2] = -sin(t)
            return dx
        end

        @test (@isdefined integ_vec)
        x0 = [0.0, 1.0]
        sol11 = taylorinteg(integ_vec, x0, 0.0, pi, _order, _abstol, parse_eqs=false,
            dense=false)
        tv11, xv11 = sol11.t, sol11.x
        sol12 = (@test_logs min_level=Logging.Warn taylorinteg(
            integ_vec, x0, 0.0, pi, _order, _abstol, dense=false))
        tv12, xv12 = sol12.t, sol12.x
            @test length(tv11) == length(tv12)
        @test iszero( norm(tv11-tv12, Inf) )
        @test iszero( norm(xv11-xv12, Inf) )

        # Compare to exact solution
        @test norm(xv12[end, 1] - sin(tv12[end]), Inf) < 1.0e-15
        @test norm(xv12[end, 2] - cos(tv12[end]), Inf) < 1.0e-15

        @taylorize function fff!(du, u, p, t)
            x, t = u
            du[1] = x
            du[2] = t
            return du
        end

        @taylorize function ggg!(du, u, p, t)
            x, t2 = u
            du[1] = x
            du[2] = t2
            return du
        end

        solF = taylorinteg(ggg!, [1.0, 0.0], 0.0, 10.0, 20, 1.0e-20, parse_eqs=false,
            dense=false)
        tvF, xvF  = solF.t, solF.x
        sol1 = taylorinteg(ggg!, [1.0, 0.0], 0.0, 10.0, 20, 1.0e-20)
        tv1, xv1  = sol1.t, sol1.x
        sol2 = taylorinteg(fff!, [1.0, 0.0], 0.0, 10.0, 20, 1.0e-20)
        tv2, xv2  = sol2.t, sol2.x

        @test tv1 == tv2 == tvF
        @test xv1 == xv2 == xvF
    end


    # Simple harmonic oscillator
    @testset "Simple harmonic oscillator" begin
        @taylorize function harm_osc!(dx, x, p, t)
            local ω = p[1]
            local ω2 = ω^2
            dx[1] = x[2]
            dx[2] = - (ω2 * x[1])
            return nothing
        end

        @test (@isdefined harm_osc!)
        q0 = [1.0, 0.0]
        p = [2.0]
        local tf = 300.0
        sol1 = taylorinteg(harm_osc!, q0, t0, tf, _order, _abstol,
            p, maxsteps=1000, parse_eqs=false, dense=false)
        tv1, xv1 = sol1.t, sol1.x
        sol1p = (@test_logs min_level=Logging.Warn taylorinteg(
            harm_osc!, q0, t0, tf, _order, _abstol, p, maxsteps=1000, dense=false))
        tv1p, xv1p = sol1p.t, sol1p.x

        @test length(tv1) == length(tv1p)
        @test iszero( norm(tv1-tv1p, Inf) )
        @test iszero( norm(xv1-xv1p, Inf) )

        # Comparing to exact solution
        @test norm(xv1p[end, 1] - cos(p[1]*tv1p[end]), Inf) < 1.0e-12
        @test norm(xv1p[end, 2] + p[1]*sin(p[1]*tv1p[end]), Inf) < 3.0e-12

        # Check that the parsed `jetcoeffs` produces the correct series in `x` and no errors
        tT = t0 + Taylor1(_order)
        qT = q0 .+ zero(tT)
        dqT = similar(qT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(harm_osc!), tT, qT, dqT, [1.0]))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(harm_osc!), tT, qT, dqT, [1.0], rv)
        @test qT[1] ≈ cos(tT)
        @test qT[2] ≈ -sin(tT)

        # The macro returns a (parsed) jetcoeffs! function which yields a `MethodError`
        # (TaylorSeries.pow! requires *two* Taylor series as first arguments) and
        # therefore it runs with the default jetcoeffs! methdo. A warning is issued.
        # To solve it, define as a local variable `ω2 = ω^2`.
        @taylorize function harm_osc_error!(dx, x, p, t)
            local ω = p[1]
            dx[1] = x[2]
            dx[2] = - (ω^2 * x[1])
            return nothing
        end

        @test !isempty(methodswith(Val{harm_osc_error!}, TI.jetcoeffs!))
        sol2e = (
            @test_logs (Warn, unable_to_parse(harm_osc_error!)) taylorinteg(
                harm_osc_error!, q0, t0, tf, _order, _abstol, p, maxsteps=1000,
                parse_eqs=true, dense=false))
        tv2e, xv2e = sol2e.t, sol2e.x
        @test length(tv1) == length(tv1p)
        @test iszero( norm(tv1-tv2e, Inf) )
        @test iszero( norm(xv1-xv2e, Inf) )

        tT = t0 + Taylor1(_order)
        qT = q0 .+ zero(tT)
        dqT = similar(qT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(harm_osc_error!), tT, qT, dqT, [1.0]))
        @test_throws MethodError  TI.jetcoeffs!(
            Val(harm_osc_error!), tT, qT, dqT, [1.0], rv)
    end

    local tf = 100.0
    @testset "Multiple pendula" begin
        NN = 3
        nnrange = 1:3
        @taylorize function multpendula1!(dx, x, p, t)
            for i in p[2]
                dx[i] = x[p[1]+i]
                dx[i+p[1]] = -sin( x[i] )
            end
            return nothing
        end

        @test (@isdefined multpendula1!)
        q0 = [pi-0.001, 0.0, pi-0.001, 0.0,  pi-0.001, 0.0]
        sol1 = taylorinteg(multpendula1!, q0, t0, tf, _order, _abstol,
            [NN, nnrange], maxsteps=1000, parse_eqs=false, dense=false)
        tv1, xv1 = sol1.t, sol1.x
        sol1p = (@test_logs min_level=Logging.Warn taylorinteg(
            multpendula1!, q0, t0, tf, _order, _abstol, [NN, nnrange], maxsteps=1000,
            dense=false))
        tv1p, xv1p = sol1p.t, sol1p.x

        @test length(tv1) == length(tv1p)
        @test iszero( norm(tv1-tv1p, Inf) )
        @test iszero( norm(xv1-xv1p, Inf) )

        @taylorize function multpendula2!(dx, x, p, t)
            local NN = 3
            local nnrange = 1:NN
            for i in nnrange
                dx[i] = x[NN+i]
                dx[i+NN] = -sin( x[i] )
            end
            return nothing
        end

        @test (@isdefined multpendula2!)
        sol2 = taylorinteg(multpendula2!, q0, t0, tf, _order, _abstol,
            maxsteps=1000, parse_eqs=false, dense=false)
        tv2, xv2 = sol2.t, sol2.x
        sol2p = (@test_logs min_level=Logging.Warn taylorinteg(
            multpendula2!, q0, t0, tf, _order, _abstol, maxsteps=1000, dense=false))
        tv2p, xv2p = sol2p.t, sol2p.x

        @test length(tv2) == length(tv2p)
        @test iszero( norm(tv2-tv2p, Inf) )
        @test iszero( norm(xv2-xv2p, Inf) )

        @taylorize function multpendula3!(dx, x, p, t)
            nn, nnrng = p   # `local` is not needed to reassign `p`;
                            # internally is treated as local)
            for i in nnrng
                dx[i] = x[nn+i]
                dx[i+nn] = -sin( x[i] )
            end
            return nothing
        end

        @test (@isdefined multpendula3!)
        sol3 = taylorinteg(multpendula3!, q0, t0, tf, _order, _abstol,
            [NN, nnrange], maxsteps=1000, parse_eqs=false, dense=false)
        tv3, xv3 = sol3.t, sol3.x
        sol3p = (@test_logs min_level=Logging.Warn taylorinteg(
            multpendula3!, q0, t0, tf, _order, _abstol, [NN, nnrange], maxsteps=1000,
            dense=false))
        tv3p, xv3p = sol3p.t, sol3p.x

        # Comparing integrations
        @test length(tv1) == length(tv2) == length(tv3)
        @test iszero( norm(tv1-tv2, Inf) )
        @test iszero( norm(tv1-tv3, Inf) )
        @test iszero( norm(xv1-xv2, Inf) )
        @test iszero( norm(xv1-xv3, Inf) )

        # Check that the parsed `jetcoeffs` produces the correct series in `x` and no errors
        tT = 0.0 + Taylor1(_order)
        qT = q0 .+ zero(tT)
        dqT = similar(qT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(multpendula1!), tT, qT, dqT, (NN, nnrange)))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(multpendula1!), tT, qT, dqT, (NN, nnrange), rv)

        qT = q0 .+ zero(tT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(multpendula2!), tT, qT, dqT, nothing))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(multpendula2!), tT, qT, dqT, nothing, rv)

        qT = q0 .+ zero(tT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(multpendula3!), tT, qT, dqT, [NN, nnrange]))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(multpendula3!), tT, qT, dqT, [NN, nnrange], rv)
    end


    # Kepler problem
    # Redefining some constants
    local _order = 28
    local tf = 2π*100.0
    @testset "Kepler problem (using `^`)" begin
        @taylorize function kepler1!(dq, q, p, t)
            local μ = p
            r_p3d2 = (q[1]^2+q[2]^2)^1.5

            dq[1] = q[3]
            dq[2] = q[4]
            dq[3] = μ * q[1]/r_p3d2
            dq[4] = μ * q[2]/r_p3d2

            return nothing
        end

        @taylorize function kepler2!(dq, q, p, t)
            nn, μ = p
            r2 = zero(q[1])
            for i = 1:nn
                r2_aux = r2 + q[i]^2
                r2 = r2_aux
            end
            r_p3d2 = r2^(3/2)
            for j = 1:nn
                dq[j] = q[nn+j]
                dq[nn+j] = μ*q[j]/r_p3d2
            end

            nothing
        end

        @test (@isdefined kepler1!)
        @test (@isdefined kepler2!)
        pars = (2, -1.0)
        q0 = [0.2, 0.0, 0.0, 3.0]
        sol1 = taylorinteg(kepler1!, q0, t0, tf, _order, _abstol, -1.0,
            maxsteps=500000, parse_eqs=false, dense=false)
        tv1, xv1 = sol1.t, sol1.x
        sol1p = (@test_logs min_level=Logging.Warn taylorinteg(kepler1!, q0, t0, tf, _order,
            _abstol, -1.0, maxsteps=500000, dense=false))
        tv1p, xv1p = sol1p.t, sol1p.x

        sol6p = taylorinteg(kepler2!, q0, t0, tf, _order, _abstol, pars,
            maxsteps=500000, parse_eqs=false, dense=false)
        tv6p, xv6p = sol6p.t, sol6p.x
        sol7p = (@test_logs min_level=Logging.Warn taylorinteg(kepler2!, q0, t0, tf, _order,
            _abstol, pars, maxsteps=500000, dense=false))
        tv7p, xv7p = sol7p.t, sol7p.x

        @test length(tv1) == length(tv1p) == length(tv6p)
        @test iszero( norm(tv1-tv1p, Inf) )
        @test iszero( norm(xv1-xv1p, Inf) )
        @test iszero( norm(tv1-tv6p, Inf) )
        @test iszero( norm(xv1-xv6p, Inf) )
        @test iszero( norm(tv7p-tv6p, Inf) )
        @test iszero( norm(xv7p-xv6p, Inf) )

        @taylorize function kepler3!(dq, q, p, t)
            local mμ = -1.0
            r_p3d2 = (q[1]^2+q[2]^2)^1.5

            dq[1] = q[3]
            dq[2] = q[4]
            dq[3] = mμ*q[1]/r_p3d2
            dq[4] = mμ*q[2]/r_p3d2

            return nothing
        end

        @taylorize function kepler4!(dq, q, p, t)
            local mμ = -1.0
            local NN = 2
            r2 = zero(q[1])
            for i = 1:NN
                r2_aux = r2 + q[i]^2
                r2 = r2_aux
            end
            r_p3d2 = r2^(3/2)
            for j = 1:NN
                dq[j] = q[NN+j]
                dq[NN+j] = mμ*q[j]/r_p3d2
            end

            nothing
        end

        @test (@isdefined kepler3!)
        @test (@isdefined kepler4!)
        sol3 = taylorinteg(kepler3!, q0, t0, tf, _order, _abstol,
            maxsteps=500000, parse_eqs=false, dense=false)
        tv3, xv3 = sol3.t, sol3.x
        sol3p = (@test_logs min_level=Logging.Warn taylorinteg(kepler3!, q0, t0, tf, _order,
            _abstol, maxsteps=500000, dense=false))
        tv3p, xv3p = sol3p.t, sol3p.x

        sol4 = taylorinteg(kepler4!, q0, t0, tf, _order, _abstol,
            maxsteps=500000, parse_eqs=false, dense=false)
        tv4, xv4 = sol4.t, sol4.x
        sol4p = (@test_logs min_level=Logging.Warn taylorinteg(kepler4!, q0, t0, tf, _order,
            _abstol, maxsteps=500000, dense=false))
        tv4p, xv4p = sol4p.t, sol4p.x

        @test length(tv3) == length(tv3p) == length(tv4)
        @test iszero( norm(tv3-tv3p, Inf) )
        @test iszero( norm(xv3-xv3p, Inf) )
        @test iszero( norm(tv3-tv4, Inf) )
        @test iszero( norm(xv3-xv4, Inf) )
        @test iszero( norm(tv4p-tv4, Inf) )
        @test iszero( norm(xv4p-xv4, Inf) )

        # Comparing both integrations
        @test iszero( norm(tv1p-tv3p, Inf) )
        @test iszero( norm(xv1p-xv3p, Inf) )

        # Check that the parsed `jetcoeffs` produces the correct series in `x`;
        # we check that it does not throws an error
        tT = 0.0 + Taylor1(_order)
        qT = q0 .+ zero(tT)
        dqT = similar(qT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(kepler1!), tT, qT, dqT, -1.0))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(kepler1!), tT, qT, dqT, -1.0, rv)

        qT = q0 .+ zero(tT)
        dqT = similar(qT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(kepler2!), tT, qT, dqT, pars))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(kepler2!), tT, qT, dqT, pars, rv)

        qT = q0 .+ zero(tT)
        dqT = similar(qT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(kepler3!), tT, qT, dqT, nothing))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(kepler3!), tT, qT, dqT, nothing, rv)

        qT = q0 .+ zero(tT)
        dqT = similar(qT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(kepler4!), tT, qT, dqT, nothing))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(kepler4!), tT, qT, dqT, nothing, rv)
    end


    @testset "Kepler problem (using `sqrt`)" begin
        @taylorize function kepler1!(dq, q, p, t)
            local μ = -1.0
            r = sqrt(q[1]^2+q[2]^2)
            r_p3d2 = r^3

            dq[1] = q[3]
            dq[2] = q[4]
            dq[3] = μ * q[1] / r_p3d2
            dq[4] = μ * q[2] / r_p3d2

            return nothing
        end

        @taylorize function kepler2!(dq, q, p, t)
            local NN = 2
            local μ = p
            r2 = zero(q[1])
            for i = 1:NN
                r2_aux = r2 + q[i]^2
                r2 = r2_aux
            end
            r = sqrt(r2)
            r_p3d2 = r^3
            for j = 1:NN
                dq[j] = q[NN+j]
                dq[NN+j] =  μ * q[j] / r_p3d2
            end

            nothing
        end

        @test (@isdefined kepler1!)
        @test (@isdefined kepler2!)
        q0 = [0.2, 0.0, 0.0, 3.0]
        sol1 = taylorinteg(kepler1!, q0, t0, tf, _order, _abstol,
            maxsteps=500000, parse_eqs=false, dense=false)
        tv1, xv1 = sol1.t, sol1.x
        sol1p = (@test_logs min_level=Logging.Warn taylorinteg(
            kepler1!, q0, t0, tf, _order, _abstol, maxsteps=500000, dense=false))
        tv1p, xv1p = sol1p.t, sol1p.x

        sol2 = taylorinteg(kepler2!, q0, t0, tf, _order, _abstol, -1.0,
            maxsteps=500000, parse_eqs=false, dense=false)
        tv2, xv2 = sol2.t, sol2.x
        sol2p = (@test_logs min_level=Logging.Warn taylorinteg(
            kepler2!, q0, t0, tf, _order, _abstol, -1.0, maxsteps=500000, dense=false))
        tv2p, xv2p = sol2p.t, sol2p.x

        @test length(tv1) == length(tv1p) == length(tv2)
        @test iszero( norm(tv1-tv1p, Inf) )
        @test iszero( norm(xv1-xv1p, Inf) )
        @test iszero( norm(tv1-tv2, Inf) )
        @test iszero( norm(xv1-xv2, Inf) )
        @test iszero( norm(tv2p-tv2, Inf) )
        @test iszero( norm(xv2p-xv2, Inf) )

        @taylorize function kepler3!(dq, q, p, t)
            local μ = p
            x, y, px, py = q
            r = sqrt(x^2+y^2)
            r_p3d2 = r^3

            dq[1] = px
            dq[2] = py
            dq[3] = μ * x / r_p3d2
            dq[4] = μ * y / r_p3d2

            return nothing
        end

        @taylorize function kepler4!(dq, q, p, t)
            local NN = 2
            local μ = -1.0
            r2 = zero(q[1])
            for i = 1:NN
                r2_aux = r2 + q[i]^2
                r2 = r2_aux
            end
            r = sqrt(r2)
            r_p3d2 = r^3
            for j = 1:NN
                dq[j] = q[NN+j]
                dq[NN+j] = μ * q[j] / r_p3d2
            end

            nothing
        end

        @test (@isdefined kepler3!)
        @test (@isdefined kepler4!)
        sol3 = taylorinteg(kepler3!, q0, t0, tf, _order, _abstol, -1.0,
            maxsteps=500000, parse_eqs=false, dense=false)
        tv3, xv3 = sol3.t, sol3.x
        sol3p = (@test_logs min_level=Logging.Warn taylorinteg(
            kepler3!, q0, t0, tf, _order, _abstol, -1.0, maxsteps=500000, dense=false))
        tv3p, xv3p = sol3p.t, sol3p.x

        sol4 = taylorinteg(kepler4!, q0, t0, tf, _order, _abstol,
            maxsteps=500000, parse_eqs=false, dense=false)
        tv4, xv4 = sol4.t, sol4.x
        sol4p = (@test_logs min_level=Logging.Warn taylorinteg(
            kepler4!, q0, t0, tf, _order, _abstol, maxsteps=500000, dense=false))
        tv4p, xv4p = sol4p.t, sol4p.x

        @test length(tv3) == length(tv3p) == length(tv4)
        @test iszero( norm(tv3-tv3p, Inf) )
        @test iszero( norm(xv3-xv3p, Inf) )
        @test iszero( norm(tv3-tv4, Inf) )
        @test iszero( norm(xv3-xv4, Inf) )
        @test iszero( norm(tv4p-tv4, Inf) )
        @test iszero( norm(xv4p-xv4, Inf) )

        # Comparing both integrations
        @test iszero( norm(tv1p-tv3p, Inf) )
        @test iszero( norm(xv1p-xv3p, Inf) )

        # Check that the parsed `jetcoeffs` produces the correct series in `x`;
        # we check that it does not throws an error
        tT = 0.0 + Taylor1(_order)
        qT = q0 .+ zero(tT)
        dqT = similar(qT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(kepler1!), tT, qT, dqT, nothing))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(kepler1!), tT, qT, dqT, nothing, rv)

        qT = q0 .+ zero(tT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(kepler2!), tT, qT, dqT, -1.0))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(kepler2!), tT, qT, dqT, -1.0, rv)

        qT = q0 .+ zero(tT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(kepler3!), tT, qT, dqT, -1.0))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(kepler3!), tT, qT, dqT, -1.0, rv)

        qT = q0 .+ zero(tT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(kepler4!), tT, qT, dqT, nothing))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(kepler4!), tT, qT, dqT, nothing, rv)
    end


    local tf = 20.0
    @testset "Lyapunov spectrum and `@taylorize`" begin
        #Lorenz system parameters
        params = [16.0, 45.92, 4.0]

        #Lorenz system ODE:
        @taylorize function lorenz1!(dq, q, p, t)
            x, y, z = q
            local σ, ρ, β = p
            dq[1] = σ*(y-x)
            dq[2] = x*(ρ-z) - y
            dq[3] = x*y - β*z
            nothing
        end

        #Lorenz system Jacobian (in-place):
        function lorenz1_jac!(jac, q, p, t)
            x, y, z = q
            σ, ρ, β = p
            jac[1,1] = -σ+zero(x)
            jac[2,1] = ρ-z
            jac[3,1] = y
            jac[1,2] = σ+zero(x)
            jac[2,2] = -one(y)
            jac[3,2] = x
            jac[1,3] = zero(z)
            jac[2,3] = -x
            jac[3,3] = -β+zero(x)
            nothing
        end

        q0 = [19.0, 20.0, 50.0] #the initial condition
        xi = set_variables("δ", order=1, numvars=length(q0))

        sol1 = lyap_taylorinteg(lorenz1!, q0, t0, tf, _order, _abstol,
            params, maxsteps=2000, parse_eqs=false)
        tv1, xv1, lv1 = sol1.t, sol1.x, sol1.λ

        solp = (@test_logs min_level=Logging.Warn lyap_taylorinteg(
            lorenz1!, q0, t0, tf, _order, _abstol, params, maxsteps=2000))
        tv1p, xv1p, lv1p = solp.t, solp.x, solp.λ

        @test tv1 == tv1p
        @test xv1 == xv1p
        @test lv1 == lv1p

        sol2 = lyap_taylorinteg(lorenz1!, q0, t0, tf, _order, _abstol,
            params, lorenz1_jac!, maxsteps=2000, parse_eqs=false)
        tv2, xv2, lv2 = sol2.t, sol2.x, sol2.λ

        sol2p = (@test_logs min_level=Logging.Warn lyap_taylorinteg(
            lorenz1!, q0, t0, tf, _order, _abstol, params, lorenz1_jac!, maxsteps=2000,
                parse_eqs=true))
        tv2p, xv2p, lv2p = sol2p.t, sol2p.x, sol2p.λ

        @test tv2 == tv2p
        @test xv2 == xv2p
        @test lv2 == lv2p

        # Comparing both integrations (lorenz1)
        @test tv1 == tv2
        @test xv1 == xv2
        @test lv1 == lv2

        #Lorenz system ODE:
        @taylorize function lorenz2!(dx, x, p, t)
            #Lorenz system parameters
            local σ = 16.0
            local β = 4.0
            local ρ = 45.92

            dx[1] = σ*(x[2]-x[1])
            dx[2] = x[1]*(ρ-x[3])-x[2]
            dx[3] = x[1]*x[2]-β*x[3]
            nothing
        end

        #Lorenz system Jacobian (in-place):
        function lorenz2_jac!(jac, x, p, t)
            #Lorenz system parameters
            local σ = 16.0
            local β = 4.0
            local ρ = 45.92

            jac[1,1] = -σ+zero(x[1])
            jac[2,1] = ρ-x[3]
            jac[3,1] = x[2]
            jac[1,2] = σ+zero(x[1])
            jac[2,2] = -1.0+zero(x[1])
            jac[3,2] = x[1]
            jac[1,3] = zero(x[1])
            jac[2,3] = -x[1]
            jac[3,3] = -β+zero(x[1])
            nothing
        end

        sol3 = lyap_taylorinteg(lorenz2!, q0, t0, tf, _order, _abstol,
            maxsteps=2000, parse_eqs=false)
        tv3, xv3, lv3 = sol3.t, sol3.x, sol3.λ

        sol3p = (@test_logs min_level=Logging.Warn lyap_taylorinteg(
            lorenz2!, q0, t0, tf, _order, _abstol, maxsteps=2000))
        tv3p, xv3p, lv3p = sol3p.t, sol3p.x, sol3p.λ

        @test tv3 == tv3p
        @test xv3 == xv3p
        @test lv3 == lv3p

        sol4 = lyap_taylorinteg(lorenz2!, q0, t0, tf, _order, _abstol,
            lorenz2_jac!, maxsteps=2000, parse_eqs=false)
        tv4, xv4, lv4 = sol4.t, sol4.x, sol4.λ

        sol4p = (@test_logs min_level=Logging.Warn lyap_taylorinteg(
            lorenz2!, q0, t0, tf, _order, _abstol, lorenz2_jac!, maxsteps=2000,
                parse_eqs=true))
        tv4p, xv4p, lv4p = sol4p.t, sol4p.x, sol4p.λ

        @test tv4 == tv4p
        @test xv4 == xv4p
        @test lv4 == lv4p

        # Comparing both integrations (lorenz2)
        @test tv3 == tv4
        @test xv3 == xv4
        @test lv3 == lv4

        # Comparing both integrations (lorenz1! vs lorenz2!)
        @test tv1 == tv3
        @test xv1 == xv3
        @test lv1 == lv3

        # Using ranges
        sol5 = lyap_taylorinteg(lorenz2!, q0, t0:0.125:tf, _order, _abstol,
            maxsteps=2000, parse_eqs=false)
        lv5, xv5 = sol5.x, sol5.λ

        sol5p = (@test_logs min_level=Logging.Warn lyap_taylorinteg(
            lorenz2!, q0, t0:0.125:tf, _order, _abstol, maxsteps=2000, parse_eqs=true))
        lv5p, xv5p = sol5p.x, sol5p.λ

        @test lv5 == lv5p
        @test xv5 == xv5p

        sol6 = lyap_taylorinteg(lorenz2!, q0, t0:0.125:tf, _order, _abstol,
            lorenz2_jac!, maxsteps=2000, parse_eqs=false)
        lv6, xv6 = sol6.x, sol6.λ

        sol6p = (@test_logs min_level=Logging.Warn lyap_taylorinteg(
            lorenz2!, q0, t0:0.125:tf, _order, _abstol, lorenz2_jac!, maxsteps=2000,
                parse_eqs=true))
        lv6p, xv6p = sol6p.x, sol6p.λ

        @test lv6 == lv6p
        @test xv6 == xv6p

        # Check that no errors are thrown
        tT = 0.0 + Taylor1(_order)
        qT = q0 .+ zero(tT)
        dqT = similar(qT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(lorenz1!), tT, qT, dqT, params))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(lorenz1!), tT, qT, dqT, params, rv)

        qT = q0 .+ zero(tT)
        dqT = similar(qT)
        rv = (@test_logs min_level=Logging.Warn TI._allocate_jetcoeffs!(
            Val(lorenz2!), tT, qT, dqT, nothing))
        @test_logs min_level=Logging.Warn TI.jetcoeffs!(
            Val(lorenz2!), tT, qT, dqT, nothing, rv)
    end


    @testset "Tests for throwing errors" begin
        # Wrong number of arguments
        ex = :(function f_p!(dx, x, p, t, y)
            dx[1] = x[2]
            dx[2] = -sin( x[1] )
        end)
        @test_throws ArgumentError TI._make_parsed_jetcoeffs(ex)

        # `&&` is not yet implemented
        ex = :(function f_p!(x, p, t)
            true && x
        end)
        @test_throws ArgumentError TI._make_parsed_jetcoeffs(ex)

        # a is not an Expr; String
        ex = :(function f_p!(x, p, t)
            "a"
        end)
        @test_throws ArgumentError TI._make_parsed_jetcoeffs(ex)

        # KeyError: key :fname not found
        ex = :(begin
            x=1
            x+x
        end)
        @test_throws KeyError TI._make_parsed_jetcoeffs(ex)

        # BoundsError; no return variable defined
        ex = :(function f_p!(x, p, t)
            local cos(t)
        end)
        @test_throws BoundsError TI._make_parsed_jetcoeffs(ex)

        # ArgumentError; .= is not yet implemented
        ex = :(function err_bbroadcasting!(Dz, z, p, t)
            Dz .= z
            nothing
        end)
        @test_throws ArgumentError TI._make_parsed_jetcoeffs(ex)

        # The macro works fine, but the `jetcoeffs!` method does not work
        # (so the integration would run using `parse_eqs=false`)
        @taylorize function harm_osc!(dx, x, p, t)
            local ω = p[1]
            # local ω2 = ω^2    # Needed this to avoid an error
            dx[1] = x[2]
            dx[2] = - (ω^2 * x[1])  # ω^2 -> ω2
            return nothing
        end
        tT = t0 + Taylor1(_order)
        qT = [1.0, 0.0] .+ zero(tT)
        parse_eqs, rv = (@test_logs (Warn, unable_to_parse(harm_osc!)) TI._determine_parsing!(
            true, harm_osc!, tT, qT, similar(qT), [2.0]))
        @test !parse_eqs
        @test_throws MethodError TI.__jetcoeffs!(Val(harm_osc!), tT, qT, similar(qT), [2.0], rv)

        @taylorize function kepler1!(dq, q, p, t)
            μ = p
            r_p3d2 = (q[1]^2+q[2]^2)^1.5

            dq[1] = q[3]
            dq[2] = q[4]
            dq[3] = μ * q[1]/r_p3d2
            dq[4] = μ * q[2]/r_p3d2

            return nothing
        end
        tT = t0 + Taylor1(_order)
        qT = [0.2, 0.0, 0.0, 3.0] .+ zero(tT)
        parse_eqs, rv = (@test_logs (Warn, unable_to_parse(kepler1!)) TI._determine_parsing!(
            true, kepler1!, tT, qT, similar(qT), -1.0))
        @test !parse_eqs
        @test_throws MethodError TI.__jetcoeffs!(Val(kepler1!), tT, qT, similar(qT), -1.0, rv)

        # Error: `@taylorize` allows only to parse up tp 5-index arrays
        ex = :(function err_arr_indx!(Dz, z, p, t)
            n = size(z,1)
            arr3 = Array{typeof(z[1])}(undef, n, 1, 1)
            arr4 = Array{typeof(z[1])}(undef, n, 1, 1, 1)
            arr5 = Array{typeof(z[1])}(undef, n, 1, 1, 1, 1)
            for i in eachindex(z)
                arr5[i,1,1,1,1] = zero(z[1])
                Dz[i] = z[i] + arr5[i,1,1,1,1]
            end
            nothing
        end)
        @test_throws ErrorException("Error: `@taylorize` allows only to parse up tp 5-index arrays") TI._make_parsed_jetcoeffs(ex)

    end


    local tf = 2π*10.0
    @testset "Jet transport with @taylorize macro" begin
        @taylorize function pendulum!(dx, x, p, t)
            dx[1] = x[2]
            dx[2] = -sin( x[1] )
            nothing
        end

        varorder = 2 #the order of the variational expansion
        p = set_variables("ξ", numvars=2, order=varorder) #TaylorN steup
        q0 = [1.3, 0.0] #the initial conditions
        q0TN = q0 + p #parametrization of a small neighbourhood around the initial conditions
        # T is the librational period == 4Elliptic.K(sin(q0[1]/2)^2)
        T = 4Elliptic.K(sin(q0[1]/2)^2) # equals 7.019250311844546
        integstep = 0.25*T #the time interval between successive evaluations of the solution vector

        #the time range
        tr = t0:integstep:T;
        #note that as called below, taylorinteg uses the parsed jetcoeffs! method by default
        solp = (@test_logs min_level=Logging.Warn taylorinteg(
            pendulum!, q0, tr, _order, _abstol, maxsteps=100))
        @test tr == solp.t
        xvp = solp.x

        # "warmup" for jet transport integration
        solTN = (@test_logs (Warn, max_iters_reached()) @inferred taylorinteg(
            pendulum!, q0TN, tr, _order, _abstol, maxsteps=1, parse_eqs=false))
        @test size(solTN.x) == (5,2)
        #jet transport integration with parsed jetcoeffs!
        solTNp = (@test_logs min_level=Logging.Warn taylorinteg(
            pendulum!, q0TN, tr, _order, _abstol, maxsteps=100))
        #jet transport integration with non-parsed jetcoeffs!
        solTN = (@test_logs min_level=Logging.Warn taylorinteg(
            pendulum!, q0TN, tr, _order, _abstol, maxsteps=100, parse_eqs=false))
        @test solTN.x == solTNp.x
        @test norm(solTNp.x[:,:]() - xvp, Inf) < 1e-15

        dq = 0.0001rand(2)
        q1 = q0 + dq
        y = (@test_logs min_level=Logging.Warn taylorinteg(
            pendulum!, q1, tr, _order, _abstol, maxsteps=100))
        y_jt = solTNp.x[:,:](dq)
        @test norm(y.x-y_jt, Inf) < 1e-11

        dq = 0.001
        t = Taylor1([0.0, 1.0], 10)
        x0T1 = q0+[0t,t]
        q1 = q0+[0.0,dq]
        sol = (@test_logs min_level=Logging.Warn taylorinteg(
            pendulum!, q1, t0, 2T, _order, _abstol, dense=false))
        tv, xv = sol.t, sol.x
        solT1 = (@test_logs min_level=Logging.Warn taylorinteg(
            pendulum!, x0T1, t0, 2T, _order, _abstol, parse_eqs=false, dense=false))
        tvT1, xvT1 = solT1.t, solT1.x
        solT1p = (@test_logs min_level=Logging.Warn taylorinteg(
            pendulum!, x0T1, t0, 2T, _order, _abstol, dense=false))
        tvT1p, xvT1p = solT1p.t, solT1p.x
        @test tvT1 == tvT1p
        @test xvT1 == xvT1p
        xv_jt = xvT1p[:,:](dq)
        @test norm(xv_jt[end,:]-xv[end,:]) < 20eps(norm(xv[end,:]))

        @taylorize function kepler1!(dq, q, p, t)
            local μ = -1.0
            r_p2 = ((q[1]^2)+(q[2]^2))
            r_p3d2 = r_p2^1.5
            dq[1] = q[3]
            dq[2] = q[4]
            newtonianCoeff = μ / r_p3d2
            dq[3] = q[1] * newtonianCoeff
            dq[4] = q[2] * newtonianCoeff
            return nothing
        end

        varorder = 2 #the order of the variational expansion
        p = set_variables("ξ", numvars=4, order=varorder) #TaylorN setup
        q0 = [0.2, 0.0, 0.0, 3.0]
        q0TN = q0 + p # JT initial condition

        sol = (@test_logs (Warn, max_iters_reached()) taylorinteg(
            kepler1!, q0TN, t0, tf, _order, _abstol, maxsteps=2, parse_eqs=false, dense=false))
        tv, xv, psol = sol.t, sol.x, sol.p
        solp = (@test_logs (Warn, max_iters_reached()) taylorinteg(
            kepler1!, q0TN, t0, tf, _order, _abstol, maxsteps=2, dense=false))
        tvp, xvp, psolp = sol.t, sol.x, sol.p
        @test tv == tvp
        @test xv == xvp
        @test psol == psolp

        sol = (@test_logs min_level=Logging.Warn taylorinteg(
            kepler1!, q0TN, t0, tf, _order, _abstol, maxsteps=3000, parse_eqs=false))
        solp = (@test_logs min_level=Logging.Warn taylorinteg(
            kepler1!, q0TN, t0, tf, _order, _abstol, maxsteps=3000))

        @test length(sol.t) == length(solp.t)
        @test sol.t == solp.t
        @test solp.t[end] == tf
        @test iszero( norm(sol.x-solp.x, Inf) )
        @test iszero( norm(sol.p-solp.p, Inf) )

        # Keplerian energy
        function visviva(x)
            v2 = x[3]^2 + x[4]^2
            r = sqrt(x[1]^2 + x[2]^2)
            return 0.5v2 - 1/r
        end

        # initial energy
        E0 = visviva(solp(t0))
        # final energy
        Ef = visviva(solp(tf))
        @test norm( E0() - Ef(), Inf ) < 1e-12

        @testset "Test _taylorinteg function barrier" begin
            _t0 = t0
            _tmax = tf
            _q0 = q0TN
            _params = nothing
            _maxsteps = 3000
            _T = eltype(_t0)
            _U = eltype(_q0)
            # Initialize the vector of Taylor1 expansions
            _dof = length(_q0)
            _t = _t0 + Taylor1( _T, _order )
            _x = Array{Taylor1{_U}}(undef, _dof)
            _dx = Array{Taylor1{_U}}(undef, _dof)
            @inbounds for i in eachindex(_q0)
                _x[i] = Taylor1( _q0[i], _order )
                _dx[i] = Taylor1( zero(_q0[i]), _order )
            end
            # Determine if specialized jetcoeffs! method exists
            __parse_eqs, __rv = TaylorIntegration._determine_parsing!(true, kepler1!, _t, _x, _dx, _params);
            # Re-initialize the Taylor1 expansions
            _t = _t0 + Taylor1( _T, _order )
            _x .= Taylor1.( _q0, _order )
            _dx .= Taylor1.( zero.(_q0), _order)
            # Allocation
            _cache_true = TaylorIntegration.init_cache(TaylorIntegration.VectorCache, Val(true), _t0, _x, _maxsteps)
            _cache_false = TaylorIntegration.init_cache(TaylorIntegration.VectorCache, Val(false), _t0, _x, _maxsteps)
            solTN = @inferred TaylorIntegration._taylorinteg!(Val(true), kepler1!, _t, _x, _dx, _q0, _t0, _tmax, _abstol, __rv, _cache_true, _params; parse_eqs=__parse_eqs, maxsteps=_maxsteps)
            solTN2 = @inferred TaylorIntegration._taylorinteg!(Val(false), kepler1!, _t, _x, _dx, _q0, _t0, _tmax, _abstol, __rv, _cache_false, _params; parse_eqs=__parse_eqs, maxsteps=_maxsteps)
            @test solTN isa TaylorSolution{typeof(_t0), eltype(_q0), ndims(solTN.x), typeof(solTN.t), typeof(solTN.x), typeof(solTN.p), Nothing, Nothing, Nothing}
            @test solTN2 isa TaylorSolution{typeof(_t0), eltype(_q0), ndims(solTN.x), typeof(solTN.t), typeof(solTN.x), Nothing, Nothing, Nothing, Nothing}
        end
    end


    @testset "Poincare maps with the pendulum" begin
        @taylorize function pendulum!(dx, x, p, t)
            dx[1] = x[2]
            dx[2] = -sin( x[1] )
            nothing
        end

        # Function defining the crossing events: the
        # surface of section is x[2]==0, x[1]>0
        function g(dx, x, params, t)
            if constant_term(x[1]) > 0
                return (true, x[2])
            else
                return (false, x[2])
            end
        end

        t0 = 0.0
        x0 = [1.3, 0.0]
        Tend = 7.019250311844546

        #warm-up lap and preliminary tests
        @test_logs (Warn, max_iters_reached()) taylorinteg(
            pendulum!, g, x0, t0, Tend, _order, _abstol, Val(false), maxsteps=1)
        @test_throws AssertionError taylorinteg(
            pendulum!, g, x0, t0, Tend, _order, _abstol, Val(false), maxsteps=1, eventorder=_order+1)

        #testing 0-th order root-finding
        sol = (@test_logs min_level=Logging.Warn taylorinteg(
            pendulum!, g, x0, t0, 3Tend, _order, _abstol, Val(false), maxsteps=1000))
        tv, xv, tvS, xvS, gvS = sol.t, sol.x, sol.tevents, sol.xevents, sol.gresids
        @test tv[1] == t0
        @test xv[1,:] == x0
        @test size(tvS) == (2,)
        @test norm(tvS-[Tend, 2Tend], Inf) < 1e-13
        @test norm(gvS, Inf) < eps()

        #testing 0-th order root-finding with time ranges/vectors
        tvr = [t0, Tend/2, Tend, 3Tend/2, 2Tend, 5Tend/2, 3Tend]
        @test_logs (Warn, max_iters_reached()) taylorinteg(
            pendulum!, g, x0, view(tvr, :), _order, _abstol, maxsteps=1)
        @test_throws AssertionError taylorinteg(
            pendulum!, g, x0, view(tvr, :), _order, _abstol, maxsteps=1, eventorder=_order+1)
        solr = (@test_logs min_level=Logging.Warn taylorinteg(
            pendulum!, g, x0, view(tvr, :), _order, _abstol, maxsteps=1000))
        xvr, tvSr, xvSr, gvSr = solr.x, solr.tevents, solr.xevents, solr.gresids
        @test xvr[1,:] == x0
        @test size(tvSr) == (2,)
        @test norm(tvSr-tvr[3:2:end-1], Inf) < 1e-13
        @test norm(tvr[3:2:end-1]-tvSr, Inf) < 1e-14
        @test norm(xvr[3:2:end-1,:]-xvSr, Inf) < 1e-14
        @test norm(gvSr[:]) < eps()
        @test norm(tvS-tvSr, Inf) < 5e-15
    end

    @testset "Tests parsing specific aspects of the expression" begin
        ex = :(
            function f!(dq::Array{T,1}, q::Array{T,1}, p, t) where {T}
                aa = my_simple_function(q, p, t)
                for i in 1:length(q)
                    if i == 1
                        dq[i] = 2q[i]
                    elseif i == 2
                        dq[i] = q[i]
                    elseif i == 3
                        dq[i] = aa
                        continue
                    else
                        dq[i] = my_complicate_function(q)
                        break
                    end
                end
                nothing
            end)

        newex1, newex2 = TI._make_parsed_jetcoeffs(ex)

        # Ignore declarations in the function
        @test newex1.args[1] == :(
            TaylorIntegration.jetcoeffs!(::Val{f!}, t::Taylor1{_T},
                q::AbstractArray{Taylor1{_S}, _N},
                dq::AbstractArray{Taylor1{_S}, _N}, p,
                __ralloc::TaylorIntegration.RetAlloc{Taylor1{_S}}) where
                    {_T <: Real, _S <: Number, _N})

        # Include not recognized functions as they appear
        @test newex1.args[2].args[2] == :(aa = __ralloc.v0[1])
        @test newex2.args[2].args[2] == :(aa = my_simple_function(q, p, t))
        @test newex1.args[2].args[3].args[2].args[2] == :(aa = my_simple_function(q, p, t))

        # Return line
        @test newex1.args[2].args[end] == :(return nothing)
        @test newex2.args[2].args[end] ==
            :(return TaylorIntegration.RetAlloc{Taylor1{_S}}([aa],
                [Array{Taylor1{_S},1}(undef, 0)],
                [Array{Taylor1{_S},2}(undef, 0, 0)],
                [Array{Taylor1{_S},3}(undef, 0, 0, 0)],
                [Array{Taylor1{_S},4}(undef, 0, 0, 0, 0)]))

        # Issue 96: deal with `elseif`s, `continue` and `break`
        ex = :(
            for i = 1:length(q)
                if i == 1
                    TaylorSeries.mul!(dq[i], 2, q[i], ord)
                else
                    if i == 2
                        TaylorSeries.identity!(dq[i], q[i], ord)
                    else
                        if i == 3
                            TaylorSeries.identity!(dq[i], aa, ord)
                            continue
                        else
                            dq[i] = my_complicate_function(q)
                            break
                        end
                    end
                end
            end)

        @test newex1.args[2].args[3].args[2].args[3] == Base.remove_linenums!(ex)

        # Throws no error
        ex = :(
            function err_arr_indx!(Dz, z, p, t)
                local n = size(z,1)  # important to include it!
                arr3 = Array{typeof(z[1])}(undef, n, 1, 1)
                arr4 = Array{typeof(z[1])}(undef, n, 1, 1, 1)
                for i in eachindex(z)
                    arr3[i,1,1] = zero(z[1])
                    arr4[i,1,1,1] = zero(z[1])
                    Dz[i] = z[i] + arr4[i,1,1,1]
                end
                nothing
            end)
        newex1, newex2 = TI._make_parsed_jetcoeffs(ex)
        @test newex1.args[2].args[2] == :(arr3 = __ralloc.v3[1])
        @test newex1.args[2].args[3] == :(arr4 = __ralloc.v4[1])
        @test newex2.args[2].args[end].args[1].args[3] == :([Array{Taylor1{_S}, 1}(undef, 0)])
        @test newex2.args[2].args[end].args[1].args[4] == :([Array{Taylor1{_S}, 2}(undef, 0, 0)])
        @test newex2.args[2].args[end].args[1].args[5] == :([arr3])
        @test newex2.args[2].args[end].args[1].args[6] == :([arr4])
    end

    @testset "Test @taylorize with @threads" begin
        @taylorize function f1!(dq, q, params, t)
            for i in 1:10
                dq[i] = q[i]
            end
            nothing
        end

        @taylorize function f1_parsed!(dq, q, params, t)
            for i in 1:10
                dq[i] = q[i]
            end
            nothing
        end

        @taylorize function f2!(dq, q, params, t)
            Threads.@threads for i in 1:10
                dq[i] = q[i]
            end
            nothing
        end

        @taylorize function f3!(dq, q, params, t)
            @threads for i in 1:10
                dq[i] = q[i]
            end
            nothing
        end

        x01 = Taylor1.(rand(10), _order)
        dx01 = similar(x01)
        t1 = Taylor1(_order)
        xaux1 = similar(x01)

        x01p = deepcopy(x01)
        dx01p = similar(x01)
        t1p = deepcopy(t1)

        x02 = deepcopy(x01)
        dx02 = similar(x01)
        t2 = deepcopy(t1)

        x03 = deepcopy(x01)
        dx03 = similar(x01)
        t3 = deepcopy(t1)

        # No error is thrown
        parse_eqs, rv = (@test_logs min_level=Warn TI._determine_parsing!(
            true, f1!, t1, x01, dx01, nothing))
        @test parse_eqs
        @test typeof(rv.v0) == Vector{Taylor1{Float64}}
        @test typeof(rv.v1) == Vector{Vector{Taylor1{Float64}}}
        parse_eqs, rv = (@test_logs min_level=Warn TI._determine_parsing!(
            true, f1_parsed!, t1p, x01p, dx01p, nothing))
        @test parse_eqs
        @test typeof(rv.v0) == Vector{Taylor1{Float64}}
        @test typeof(rv.v1) == Vector{Vector{Taylor1{Float64}}}
        parse_eqs, rv = (@test_logs min_level=Warn TI._determine_parsing!(
            true, f2!, t2, x02, dx02, nothing))
        @test parse_eqs
        @test typeof(rv.v0) == Vector{Taylor1{Float64}}
        @test typeof(rv.v1) == Vector{Vector{Taylor1{Float64}}}
        parse_eqs, rv = (@test_logs min_level=Warn TI._determine_parsing!(
            true, f3!, t3, x03, dx03, nothing))
        @test parse_eqs
        @test typeof(rv.v0) == Vector{Taylor1{Float64}}
        @test typeof(rv.v1) == Vector{Vector{Taylor1{Float64}}}

        @test x01 == x01p
        @test x01p == x02
        @test x02 == x03

        # An example involving coupled dofs: a 1D chain of harmonic oscillators
        @taylorize function harmosc1dchain!(dq, q, params, t)
            local N = Int(length(q)/2)
            local _eltype_q_ = eltype(q)
            local μ = params
            X = Array{_eltype_q_}(undef, N, N)
            accX = Array{_eltype_q_}(undef, N) #acceleration
            for j in 1:N
                accX[j] = zero(q[1])
                dq[j] = q[N+j]
            end
            #compute accelerations
            for j in 1:N
                for i in 1:N
                    if i == j
                    else
                        X[i,j] = q[i]-q[j]
                        temp_001 = accX[j] + (μ[i]*X[i,j])
                        accX[j] = temp_001
                    end #if i != j
                end #for, i
            end #for, j
            for i in 1:N
                dq[N+i] = accX[i]
            end
            nothing
        end

        # Same as harmosc1dchain!, but adding a `@threads for`
        @taylorize function harmosc1dchain_threads!(dq, q, params, t)
            local N = Int(length(q)/2)
            local _eltype_q_ = eltype(q)
            local μ = params
            X = Array{_eltype_q_}(undef, N, N)
            accX = Array{_eltype_q_}(undef, N) #acceleration
            for j in 1:N
                accX[j] = zero(q[1])
                dq[j] = q[N+j]
            end
            #compute accelerations
            Threads.@threads for j in 1:N
                for i in 1:N
                    if i == j
                    else
                        X[i,j] = q[i]-q[j]
                        temp_001 = accX[j] + (μ[i]*X[i,j])
                        accX[j] = temp_001
                    end #if i != j
                end #for, i
            end #for, j
            for i in 1:N
                dq[N+i] = accX[i]
            end
            nothing
        end

        N = 200
        x0 = 10randn(2N)
        t = Taylor1(_order)
        μ = 1e-7rand(N)
        x = Taylor1.(x0, t.order)
        dx = similar(x)
        t_ = deepcopy(t)
        x_ = Taylor1.(x0, t.order)
        dx_ = similar(x_)

        @show Threads.nthreads()

        parse_eqs, rv = (@test_logs min_level=Warn TI._determine_parsing!(
            true, harmosc1dchain!, t, x, dx, μ))
        @test parse_eqs
        parse_eqs, rv = (@test_logs min_level=Warn TI._determine_parsing!(
            true, harmosc1dchain_threads!, t, x_, dx_, μ))
        @test parse_eqs

        @test x == x_
        @test dx == dx_

        sol = (@test_logs min_level=Warn taylorinteg(
            harmosc1dchain!, x0, t0, 10000.0, _order, _abstol, μ))
        tv, xv = sol.t, sol.x
        sol_ = (@test_logs min_level=Warn taylorinteg(
            harmosc1dchain_threads!, x0, t0, 10000.0, _order, _abstol, μ))
        tv_, xv_ = sol_.t, sol_.x

        @test tv == tv_
        @test xv == xv_
    end

    # Issue 106: allow calls to macro from Julia packages
    @testset "Test @taylorize use in modules/packages" begin
        # TestPkg is a local (unregistered) Julia package which tests the use
        # of @taylorize inside module TestPkg. We test the direct use of
        # @taylorize, as well as _make_parsed_jetcoeffs, to check that
        # everything is compiled fine. Finally, we check that the parsed
        # jetcoeffs! expressions (nex1, nex2, nex3) generated from inside Test Pkg
        # are equivalent to (nex_1, nex_2, nex_3) generated here
        Pkg.develop(  Pkg.PackageSpec( path=joinpath(@__DIR__, "TestPkg") )  )
        using TestPkg
        nex1_, nall1_ = TI._make_parsed_jetcoeffs(TestPkg.ex1)
        nex2_, nall2_ = TI._make_parsed_jetcoeffs(TestPkg.ex2)
        nex3_, nall3_ = TI._make_parsed_jetcoeffs(TestPkg.ex3)
        @test length(TestPkg.nex1.args[2].args) == length(nex1_.args[2].args)
        @test length(TestPkg.nex2.args[2].args) == length(nex2_.args[2].args)
        @test length(TestPkg.nex3.args[2].args) == length(nex3_.args[2].args)
        Pkg.rm("TestPkg")
    end
end
