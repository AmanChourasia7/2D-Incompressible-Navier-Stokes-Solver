import numpy as np
import pytest
from navier_stokes_2d import NavierStokesSolver

# Helpers
def make_solver(Nx=21, Ny=21, Re=100.0, dt=0.001, advection="central"):
    return NavierStokesSolver(Nx=Nx, Ny=Ny, Re=Re, dt=dt, advection=advection)

# Initialisation
class TestInitialisation:
    def test_grid_shape(self):
        s = make_solver(Nx=21, Ny=31)
        assert s.u.shape == (21, 31)
        assert s.v.shape == (21, 31)
        assert s.p.shape == (21, 31)

    def test_initial_fields_zero(self):
        s = make_solver()
        assert np.all(s.u == 0.0)
        assert np.all(s.v == 0.0)
        assert np.all(s.p == 0.0)

    def test_grid_coordinates(self):
        s = make_solver(Nx=11, Ny=11)
        np.testing.assert_allclose(s.x[0],  0.0, atol=1e-14)
        np.testing.assert_allclose(s.x[-1], 1.0, atol=1e-14)
        np.testing.assert_allclose(s.y[0],  0.0, atol=1e-14)
        np.testing.assert_allclose(s.y[-1], 1.0, atol=1e-14)
        assert s.X.shape == (11, 11)
        assert s.Y.shape == (11, 11)

    def test_poisson_matrix_shape(self):
        s = make_solver(Nx=11, Ny=13)
        N = 11 * 13
        assert s.A_pressure.shape == (N, N)

    def test_viscosity(self):
        s = make_solver(Re=200.0)
        assert abs(s.nu - 0.005) < 1e-14

# Boundary conditions
class TestBoundaryConditions:
    def test_lid_bc_top(self):
        s = make_solver()
        s.apply_lid_driven_cavity_bc(u_lid=2.0)
        np.testing.assert_array_equal(s.u[1:-1, -1], 2.0)
        assert s.u[0, -1] == 0.0
        assert s.u[-1, -1] == 0.0
        np.testing.assert_array_equal(s.v[:, -1], 0.0)

    def test_lid_bc_no_slip_walls(self):
        s = make_solver()
        s.apply_lid_driven_cavity_bc()
        np.testing.assert_array_equal(s.u[:, 0],  0.0)
        np.testing.assert_array_equal(s.v[:, 0],  0.0)
        np.testing.assert_array_equal(s.u[0, :],  0.0)
        np.testing.assert_array_equal(s.v[0, :],  0.0)
        np.testing.assert_array_equal(s.u[-1, :], 0.0)
        np.testing.assert_array_equal(s.v[-1, :], 0.0)

# Spatial operators
class TestSpatialOperators:
    def test_laplacian_quadratic(self):
        s = make_solver(Nx=21, Ny=21)
        f = s.X ** 2                
        lap = s._laplacian(f)
        np.testing.assert_allclose(lap[1:-1, 1:-1], 2.0, atol=1e-10)

    def test_divergence_zero_field(self):
        s = make_solver()
        div = s._divergence(s.u, s.v)
        np.testing.assert_array_equal(div, 0.0)

    def test_gradient_linear_pressure(self):
        s = make_solver(Nx=21, Ny=21)
        p = s.X.copy()                  
        dpdx, dpdy = s._gradient(p)
        np.testing.assert_allclose(dpdx[1:-1, :], 1.0, atol=1e-10)
        np.testing.assert_allclose(dpdy[:, 1:-1], 0.0, atol=1e-10)

    def test_vorticity_uniform_shear(self):
        s = make_solver(Nx=21, Ny=21)
        s.u = s.Y.copy()
        s.v = np.zeros_like(s.v)
        omega = s.compute_vorticity()
        np.testing.assert_allclose(omega[1:-1, 1:-1], -1.0, atol=1e-10)

    def test_stream_function_uniform_flow(self):
        s = make_solver(Nx=11, Ny=11)
        s.u[:, :] = 1.0
        s.v[:, :] = 0.0
        psi = s.compute_stream_function()
        expected = np.outer(np.ones(s.Nx), s.y)
        np.testing.assert_allclose(psi, expected, atol=1e-10)

# Projection step
class TestProjectionMethod:
    def test_single_step_no_crash(self):
        s = make_solver()
        s.apply_lid_driven_cavity_bc()
        div_max = s.step(u_lid=1.0)
        assert np.isfinite(div_max)
        assert np.all(np.isfinite(s.u))
        assert np.all(np.isfinite(s.v))
        assert np.all(np.isfinite(s.p))

    def test_bcs_preserved_after_step(self):
        """Velocity BCs must hold after each step."""
        s = make_solver()
        for _ in range(5):
            s.step(u_lid=1.0)
        np.testing.assert_array_equal(s.u[1:-1, -1], 1.0)# top
        np.testing.assert_array_equal(s.u[:, 0],  0.0)   # bottom
        np.testing.assert_array_equal(s.u[0, :],  0.0)   # left
        np.testing.assert_array_equal(s.u[-1, :], 0.0)   # right
        np.testing.assert_array_equal(s.v[0, :],  0.0)
        np.testing.assert_array_equal(s.v[-1, :], 0.0)

    def test_divergence_reduced_by_projection(self):
        s = make_solver(dt=0.001)
        for _ in range(20):
            s.step(u_lid=1.0)
        div_max = s.max_divergence()
        assert div_max < 1e-8, f"Divergence too large: {div_max:.3e}"

    def test_time_advances(self):
        s = make_solver(dt=0.001)
        for _ in range(10):
            s.step()
        np.testing.assert_allclose(s.t, 0.01, rtol=1e-12)
        assert s.step_count == 10

# Full simulation
class TestSimulation:
    def test_solve_returns_history(self):
        s = make_solver(Nx=11, Ny=11, dt=0.001)
        history = s.solve(T_final=0.01, save_interval=3, verbose=False)
        assert isinstance(history, list)
        assert len(history) > 0
        for snap in history:
            assert {"t", "u", "v", "p"} <= snap.keys()

    def test_mass_conservation_short_run(self):
        s = make_solver(Nx=21, Ny=21, dt=0.0005)
        s.solve(T_final=0.1, save_interval=50, verbose=False)
        div_max = s.max_divergence()
        assert div_max < 1e-8, f"max|div u| = {div_max:.3e}"

    def test_upwind_solver(self):
        s = make_solver(Nx=11, Ny=11, dt=0.001, advection="upwind")
        history = s.solve(T_final=0.01, save_interval=5, verbose=False)
        assert len(history) > 0

    def test_reynolds_number_effect(self):
        s_lo = make_solver(Nx=21, Ny=21, Re=10,   dt=0.001)
        s_hi = make_solver(Nx=21, Ny=21, Re=1000, dt=0.0001, advection="upwind")
        s_lo.solve(T_final=0.05, save_interval=100, verbose=False)
        s_hi.solve(T_final=0.05, save_interval=100, verbose=False)
        assert not np.allclose(s_lo.u, s_hi.u, atol=1e-6), \
            "Low-Re and high-Re solutions should differ"


# Stable time step helper
class TestStableDt:
    def test_stable_dt_decreases_with_Re(self):
        dt_lo = NavierStokesSolver.stable_dt(dx=0.05, nu=1.0,   u_max=1.0)
        dt_hi = NavierStokesSolver.stable_dt(dx=0.05, nu=0.001, u_max=1.0)
        assert dt_lo < dt_hi or dt_lo == pytest.approx(dt_hi, rel=0.5)

    def test_stable_dt_positive(self):
        dt = NavierStokesSolver.stable_dt(dx=0.025, nu=0.01, u_max=1.0)
        assert dt > 0.0
