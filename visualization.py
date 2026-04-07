import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_velocity_magnitude(solver, ax=None, cmap="viridis", title=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    speed = np.sqrt(solver.u ** 2 + solver.v ** 2)
    im = ax.contourf(solver.X.T, solver.Y.T, speed.T, levels=20, cmap=cmap)
    plt.colorbar(im, ax=ax, label="|u|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Velocity Magnitude")
    ax.set_aspect("equal")
    return fig


def plot_streamlines(solver, ax=None, density=1.5, cmap="coolwarm", title=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    speed = np.sqrt(solver.u ** 2 + solver.v ** 2)

    strm = ax.streamplot(
        solver.x, solver.y,
        solver.u.T, solver.v.T,
        color=speed.T,
        cmap=cmap,
        density=density,
        linewidth=0.8,
    )
    plt.colorbar(strm.lines, ax=ax, label="|u|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Streamlines")
    ax.set_aspect("equal")
    return fig


def plot_vorticity(solver, ax=None, cmap="RdBu_r", title=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    omega = solver.compute_vorticity()
    vmax = max(float(np.abs(omega).max()), 1e-12)

    im = ax.contourf(
        solver.X.T, solver.Y.T, omega.T,
        levels=20, cmap=cmap, vmin=-vmax, vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label="ω = ∂v/∂x − ∂u/∂y")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Vorticity")
    ax.set_aspect("equal")
    return fig


def plot_pressure(solver, ax=None, cmap="jet", title=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    im = ax.contourf(solver.X.T, solver.Y.T, solver.p.T, levels=20, cmap=cmap)
    plt.colorbar(im, ax=ax, label="p")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Pressure")
    ax.set_aspect("equal")
    return fig


def plot_stream_function(solver, ax=None, cmap="plasma", title=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    psi = solver.compute_stream_function()
    im = ax.contourf(solver.X.T, solver.Y.T, psi.T, levels=20, cmap=cmap)
    plt.colorbar(im, ax=ax, label="ψ")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Stream Function")
    ax.set_aspect("equal")
    return fig


def plot_centerline_velocities(solver, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x_mid = solver.Nx // 2
    y_mid = solver.Ny // 2

    u_cl = solver.u[x_mid, :]      
    v_cl = solver.v[:, y_mid]     

    ax.plot(solver.y, u_cl, "b-",  label="u(x=0.5, y)")
    ax.plot(solver.x, v_cl, "r--", label="v(x, y=0.5)")
    ax.axhline(0.0, color="k", linestyle=":", linewidth=0.6)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(title or "Centerline Velocities")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_divergence_history(div_history, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.semilogy(div_history, "b-", linewidth=0.8)
    ax.set_xlabel("Time step")
    ax.set_ylabel("max |∇·u|")
    ax.set_title("Divergence Residual (Mass Conservation)")
    ax.grid(True, which="both", alpha=0.3)
    return fig


def plot_summary(solver, Re=None, figsize=(13, 10)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    title = "Lid-Driven Cavity Flow"
    if Re is not None:
        title += f",  Re = {Re}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    plot_velocity_magnitude(solver, ax=axes[0, 0])
    plot_streamlines(solver, ax=axes[0, 1])
    plot_vorticity(solver, ax=axes[1, 0])
    plot_pressure(solver, ax=axes[1, 1])

    plt.tight_layout()
    return fig
