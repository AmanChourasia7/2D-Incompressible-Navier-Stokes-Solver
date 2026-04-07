import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


class NavierStokesSolver:

    def __init__(
        self,
        Nx: int = 41,
        Ny: int = 41,
        Lx: float = 1.0,
        Ly: float = 1.0,
        Re: float = 100.0,
        dt: float = 0.001,
        advection: str = "central",
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.Re = Re
        self.nu = 1.0 / Re          # kinematic viscosity (normalised)
        self.dt = dt
        self.advection = advection

        # Grid spacing
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)

        # Grid coordinates
        self.x = np.linspace(0.0, Lx, Nx)
        self.y = np.linspace(0.0, Ly, Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # Velocity and pressure fields
        self.u = np.zeros((Nx, Ny))   # x-velocity
        self.v = np.zeros((Nx, Ny))   # y-velocity
        self.p = np.zeros((Nx, Ny))   # pressure

        # Simulation time and step counter
        self.t = 0.0
        self.step_count = 0

        # Preassemble the pressure Poisson matrix (which are reused every step)
        self._build_poisson_matrix()

    # Grid operators
    @staticmethod
    def _laplacian_1d_neumann(N: int, h: float) -> sparse.csr_matrix:
        diag = np.full(N, -2.0 / h ** 2)
        off = np.ones(N - 1) / h ** 2

        L = sparse.diags([off, diag, off], [-1, 0, 1],
                         shape=(N, N), format="lil")
        # Double the off diagonal at the boundary rows to account for ghost
        L[0, 1] = 2.0 / h ** 2
        L[-1, -2] = 2.0 / h ** 2
        return L.tocsr()

    def _build_poisson_matrix(self) -> None:
        Lx_1d = self._laplacian_1d_neumann(self.Nx, self.dx)
        Ly_1d = self._laplacian_1d_neumann(self.Ny, self.dy)

        A = (sparse.kron(Lx_1d, sparse.eye(self.Ny)) +
             sparse.kron(sparse.eye(self.Nx), Ly_1d))

        # Pin p[0, 0] = 0 to make the system non singular
        A = A.tolil()
        A[0, :] = 0.0
        A[0, 0] = 1.0
        self.A_pressure = A.tocsr()

    # Spatial derivative helpers
    def _laplacian(self, f: np.ndarray) -> np.ndarray:
        """Second-order central-difference Laplacian (interior only)."""
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1] = (
            (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / self.dx ** 2
            + (f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1] + f[1:-1, :-2]) / self.dy ** 2
        )
        return lap

    def _advection_central(
        self, u: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = self.dx, self.dy

        dudx = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx)
        dudy = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dy)
        dvdx = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dx)
        dvdy = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dy)

        adv_u = np.zeros_like(u)
        adv_v = np.zeros_like(v)
        adv_u[1:-1, 1:-1] = u[1:-1, 1:-1] * dudx + v[1:-1, 1:-1] * dudy
        adv_v[1:-1, 1:-1] = u[1:-1, 1:-1] * dvdx + v[1:-1, 1:-1] * dvdy
        return adv_u, adv_v

    def _advection_upwind(
        self, u: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = self.dx, self.dy
        adv_u = np.zeros_like(u)
        adv_v = np.zeros_like(v)

        ui = u[1:-1, 1:-1]
        vi = v[1:-1, 1:-1]

        # Upwind in x
        dudx_bwd = (u[1:-1, 1:-1] - u[:-2, 1:-1]) / dx
        dudx_fwd = (u[2:, 1:-1] - u[1:-1, 1:-1]) / dx
        dvdx_bwd = (v[1:-1, 1:-1] - v[:-2, 1:-1]) / dx
        dvdx_fwd = (v[2:, 1:-1] - v[1:-1, 1:-1]) / dx

        # Upwind in y
        dudy_bwd = (u[1:-1, 1:-1] - u[1:-1, :-2]) / dy
        dudy_fwd = (u[1:-1, 2:] - u[1:-1, 1:-1]) / dy
        dvdy_bwd = (v[1:-1, 1:-1] - v[1:-1, :-2]) / dy
        dvdy_fwd = (v[1:-1, 2:] - v[1:-1, 1:-1]) / dy

        adv_u[1:-1, 1:-1] = (
            np.maximum(ui, 0.0) * dudx_bwd + np.minimum(ui, 0.0) * dudx_fwd
            + np.maximum(vi, 0.0) * dudy_bwd + np.minimum(vi, 0.0) * dudy_fwd
        )
        adv_v[1:-1, 1:-1] = (
            np.maximum(ui, 0.0) * dvdx_bwd + np.minimum(ui, 0.0) * dvdx_fwd
            + np.maximum(vi, 0.0) * dvdy_bwd + np.minimum(vi, 0.0) * dvdy_fwd
        )
        return adv_u, adv_v

    def _divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        dx, dy = self.dx, self.dy
        div = np.zeros((self.Nx, self.Ny))

        # Interior
        div[1:-1, 1:-1] = (
            (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx)
            + (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dy)
        )
        # Boundary edges
        div[0, 1:-1] = (
            (u[1, 1:-1] - u[0, 1:-1]) / dx
            + (v[0, 2:] - v[0, :-2]) / (2.0 * dy)
        )
        div[-1, 1:-1] = (
            (u[-1, 1:-1] - u[-2, 1:-1]) / dx
            + (v[-1, 2:] - v[-1, :-2]) / (2.0 * dy)
        )
        div[1:-1, 0] = (
            (u[2:, 0] - u[:-2, 0]) / (2.0 * dx)
            + (v[1:-1, 1] - v[1:-1, 0]) / dy
        )
        div[1:-1, -1] = (
            (u[2:, -1] - u[:-2, -1]) / (2.0 * dx)
            + (v[1:-1, -1] - v[1:-1, -2]) / dy
        )
        return div

    def _divergence_fwd(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:

        dx, dy = self.dx, self.dy
        div = np.zeros((self.Nx, self.Ny))

        # Forward differences for all but the last row/coloumn
        div[:-1, :-1] = (
            (u[1:, :-1] - u[:-1, :-1]) / dx
            + (v[:-1, 1:] - v[:-1, :-1]) / dy
        )
        # Right boundary is backward in x
        div[-1, :-1] = (
            (u[-1, :-1] - u[-2, :-1]) / dx
            + (v[-1, 1:] - v[-1, :-1]) / dy
        )
        # Top boundary is backward in y
        div[:-1, -1] = (
            (u[1:, -1] - u[:-1, -1]) / dx
            + (v[:-1, -1] - v[:-1, -2]) / dy
        )
        # Top-right corner
        div[-1, -1] = (
            (u[-1, -1] - u[-2, -1]) / dx
            + (v[-1, -1] - v[-1, -2]) / dy
        )
        return div

    def _gradient(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = self.dx, self.dy
        dpdx = np.zeros_like(p)
        dpdy = np.zeros_like(p)

        # Interior
        dpdx[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2.0 * dx)
        dpdy[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * dy)
        # Boundary
        dpdx[0, :] = (p[1, :] - p[0, :]) / dx
        dpdx[-1, :] = (p[-1, :] - p[-2, :]) / dx
        dpdy[:, 0] = (p[:, 1] - p[:, 0]) / dy
        dpdy[:, -1] = (p[:, -1] - p[:, -2]) / dy
        return dpdx, dpdy

    def _gradient_bwd(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = self.dx, self.dy
        dpdx = np.zeros_like(p)
        dpdy = np.zeros_like(p)

        # Backward in x for all but the left boundary
        dpdx[1:, :] = (p[1:, :] - p[:-1, :]) / dx
        dpdx[0, :] = (p[1, :] - p[0, :]) / dx       # forward at left boundary
        # Backward in y for all but the bottom boundary
        dpdy[:, 1:] = (p[:, 1:] - p[:, :-1]) / dy
        dpdy[:, 0] = (p[:, 1] - p[:, 0]) / dy       # forward at bottom boundary
        return dpdx, dpdy

    # Boundary-condition helpers
    def apply_lid_driven_cavity_bc(self, u_lid: float = 1.0) -> None:
        # Top lid
        self.u[:, -1] = u_lid
        self.v[:, -1] = 0.0
        # Bottom wall
        self.u[:, 0] = 0.0
        self.v[:, 0] = 0.0
        # Left wall
        self.u[0, :] = 0.0
        self.v[0, :] = 0.0
        # Right wall
        self.u[-1, :] = 0.0
        self.v[-1, :] = 0.0

    @staticmethod
    def _apply_cavity_bc_array(
        u: np.ndarray, v: np.ndarray, u_lid: float
    ) -> None:
        u[:, -1] = u_lid;  v[:, -1] = 0.0
        u[:, 0]  = 0.0;    v[:, 0]  = 0.0
        u[0, :]  = 0.0;    v[0, :]  = 0.0
        u[-1, :] = 0.0;    v[-1, :] = 0.0

    # Time stepping (projection method)
    def step(self, u_lid: float = 1.0) -> float:
        u, v, dt, nu = self.u, self.v, self.dt, self.nu

        # predictor
        if self.advection == "upwind":
            adv_u, adv_v = self._advection_upwind(u, v)
        else:
            adv_u, adv_v = self._advection_central(u, v)

        lap_u = self._laplacian(u)
        lap_v = self._laplacian(v)

        u_star = u + dt * (-adv_u + nu * lap_u)
        v_star = v + dt * (-adv_v + nu * lap_v)

        # Enforce velocity BCs on intermediate field
        self._apply_cavity_bc_array(u_star, v_star, u_lid)


        # pressure Poissons
        div_star = self._divergence_fwd(u_star, v_star)
        rhs_flat = (1.0 / dt) * div_star.ravel()
        rhs_flat[0] = 0.0          # enforce p[0, 0] = 0 (removes null space)

        p_new = spsolve(self.A_pressure, rhs_flat).reshape(self.Nx, self.Ny)

        # corrector
        dpdx, dpdy = self._gradient_bwd(p_new)

        u_new = u_star - dt * dpdx
        v_new = v_star - dt * dpdy

        # Re-apply velocity BCs (projection slightly perturbs boundary values)
        self._apply_cavity_bc_array(u_new, v_new, u_lid)

        # Update state
        self.u = u_new
        self.v = v_new
        self.p = p_new
        self.t += dt
        self.step_count += 1

        div_max = float(np.max(np.abs(self._divergence_fwd(u_new, v_new)[2:-2, 2:-2])))
        return div_max

    def solve(
        self,
        T_final: float,
        u_lid: float = 1.0,
        save_interval: int = 100,
        verbose: bool = True,
    ) -> list[dict]:
        n_steps = max(1, int(round(T_final / self.dt)))
        history: list[dict] = []

        self.apply_lid_driven_cavity_bc(u_lid)

        for n in range(n_steps):
            div_max = self.step(u_lid)

            if n % save_interval == 0:
                history.append(
                    dict(t=self.t, u=self.u.copy(), v=self.v.copy(), p=self.p.copy())
                )
                if verbose:
                    u_max = float(np.max(np.abs(self.u)))
                    print(
                        f"  step {n:6d}/{n_steps}  t={self.t:.4f}"
                        f"  |div u|_max={div_max:.2e}  |u|_max={u_max:.4f}"
                    )

        return history

    # Derived quantity helpers
    def compute_vorticity(self) -> np.ndarray:
        dx, dy = self.dx, self.dy
        omega = np.zeros((self.Nx, self.Ny))
        omega[1:-1, 1:-1] = (
            (self.v[2:, 1:-1] - self.v[:-2, 1:-1]) / (2.0 * dx)
            - (self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2.0 * dy)
        )
        return omega

    def compute_stream_function(self) -> np.ndarray:
        psi = np.zeros((self.Nx, self.Ny))
        for j in range(1, self.Ny):
            psi[:, j] = psi[:, j - 1] + self.u[:, j] * self.dy
        return psi

    def max_divergence(self) -> float:
        return float(np.max(np.abs(self._divergence_fwd(self.u, self.v)[2:-2, 2:-2])))

    # Stability helpers

    @classmethod
    def stable_dt(
        cls,
        dx: float,
        dy: float = None,
        nu: float = 0.01,
        u_max: float = 1.0,
        cfl: float = 0.25,
    ) -> float:
        if dy is None:
            dy = dx
        dt_adv = cfl * min(dx, dy) / (u_max + 1e-12)
        dt_diff = cfl * min(dx, dy) ** 2 / (2.0 * nu) if nu > 0 else float("inf")
        return min(dt_adv, dt_diff)
