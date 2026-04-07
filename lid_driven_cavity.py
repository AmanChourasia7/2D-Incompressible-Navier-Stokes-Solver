import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navier_stokes_2d import NavierStokesSolver
from visualization import plot_summary, plot_centerline_velocities


# Here is the Core simulation driver
def run_lid_driven_cavity(
    Re: float = 100.0,
    Nx: int = 41,
    Ny: int = 41,
    T_final: float = 10.0,
    dt: float = None,
    u_lid: float = 1.0,
    advection: str = "central",
    save_interval: int = 500,
    verbose: bool = True,
) -> tuple:
    
    nu = 1.0 / Re
    dx = 1.0 / (Nx - 1)
    dy = 1.0 / (Ny - 1)

    if dt is None:
        dt = NavierStokesSolver.stable_dt(dx=dx, dy=dy, nu=nu, u_max=u_lid, cfl=0.25)

    if verbose:
        print(f"\nLid-driven cavity is:  Re={Re},  Nx={Nx}×{Ny},  dt={dt:.4e},  T={T_final}")

    solver = NavierStokesSolver(
        Nx=Nx, Ny=Ny, Re=Re, dt=dt, advection=advection
    )
    history = solver.solve(
        T_final=T_final,
        u_lid=u_lid,
        save_interval=save_interval,
        verbose=verbose,
    )
    return solver, history


# Here are the Parametric studies
def parameter_study_reynolds(
    Re_list: list,
    Nx: int = 41,
    Ny: int = 41,
    T_final: float = 15.0,
    output_dir: str = "results",
    verbose: bool = True,
) -> dict:

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for Re in Re_list:
        print(f"\n{'='*55}\n  Re = {Re}\n{'='*55}")

        # Higher Re → use upwind advection for stability
        adv = "upwind" if Re >= 500 else "central"

        solver, _ = run_lid_driven_cavity(
            Re=Re, Nx=Nx, Ny=Ny, T_final=T_final,
            advection=adv, verbose=verbose,
        )
        results[Re] = solver

        fig = plot_summary(solver, Re=Re)
        out_path = os.path.join(output_dir, f"cavity_Re{int(Re):04d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")

    return results


def grid_refinement_study(
    Re: float = 100.0,
    grids: list = None,
    T_final: float = 10.0,
    output_dir: str = "results",
    verbose: bool = True,
) -> dict:

    if grids is None:
        grids = [(21, 21), (41, 41), (81, 81)]

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    fig, axes = plt.subplots(1, len(grids), figsize=(5 * len(grids), 4))
    if len(grids) == 1:
        axes = [axes]

    for ax, (Nx, Ny) in zip(axes, grids):
        print(f"\nGrid refinement:  Re={Re},  Nx={Nx}×{Ny}")
        solver, _ = run_lid_driven_cavity(
            Re=Re, Nx=Nx, Ny=Ny, T_final=T_final, verbose=verbose
        )
        results[(Nx, Ny)] = solver

        from visualization import plot_streamlines
        plot_streamlines(solver, ax=ax, density=1.2,
                         title=f"{Nx}x{Ny} grid")

    fig.suptitle(f"Grid Refinement Study  (Re = {Re})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"grid_refinement_Re{int(Re):04d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {out_path}")
    return results


# CLI entry point
def _parse_args():
    p = argparse.ArgumentParser(
        description="Lid-driven cavity flow  (2D incompressible NS solver)"
    )
    p.add_argument("--Re",     type=float, default=100.0, help="Reynolds number")
    p.add_argument("--Nx",     type=int,   default=41,    help="Grid points in x")
    p.add_argument("--Ny",     type=int,   default=41,    help="Grid points in y")
    p.add_argument("--T",      type=float, default=10.0,  help="End time")
    p.add_argument("--study",  action="store_true",       help="Run Re parameter study")
    p.add_argument("--refine", action="store_true",       help="Run grid refinement study")
    p.add_argument("--outdir", default="results",         help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.study:
        print("Reynolds-number parameter study …")
        parameter_study_reynolds(
            Re_list=[100, 400, 1000],
            Nx=args.Nx, Ny=args.Ny, T_final=args.T,
            output_dir=args.outdir,
        )

    elif args.refine:
        print("Grid-refinement study …")
        grid_refinement_study(
            Re=args.Re, T_final=args.T, output_dir=args.outdir
        )

    else:
        print(f"Single run at Re={args.Re} …")
        solver, history = run_lid_driven_cavity(
            Re=args.Re, Nx=args.Nx, Ny=args.Ny, T_final=args.T,
        )

        # Summary plots
        fig = plot_summary(solver, Re=args.Re)
        path = os.path.join(args.outdir, f"cavity_Re{int(args.Re):04d}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {path}")

        # Centreline velocity profile
        fig2 = plot_centerline_velocities(solver)
        path2 = os.path.join(args.outdir, f"centreline_Re{int(args.Re):04d}.png")
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved to {path2}")
