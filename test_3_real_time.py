"""Generate Fig. 3: resonant photon-assisted transport through three dots."""
from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
import warnings

import numpy as np
import scipy.linalg as la
from scipy.optimize import least_squares
import torch

from test_2_ground_state import METHODS, LABELS, LINESTYLES, plt, write_csv

DT, DURATION = 0.1, 10.0
GAMMA, KAPPA = 0.1, 0.005
EPSILON_2, U, U_TILDE = -0.5, 20.0, 18.0
PHOTON_CUTOFF = 4
CT, RT = torch.complex128, torch.float64

torch.set_num_threads(1)


def fermions():
    operators = []
    for i in range(3):
        a = np.zeros((8, 8), complex)
        for state in range(8):
            if (state >> i) & 1:
                a[state ^ (1 << i), state] = (-1) ** (state & ((1 << i) - 1)).bit_count()
        operators.append(a)
    return operators


D = fermions()
N = [a.conj().T @ a for a in D]
IDENTITY = np.eye(8)
# Row-major density matrices, restricted to equal bra/ket electron number.
BLOCKS = np.array([a * 8 + b for a in range(8) for b in range(8)
                   if a.bit_count() == b.bit_count()])
TRACE = IDENTITY.ravel()[BLOCKS]
LEFT = np.stack([np.diag(np.kron(n, IDENTITY))[BLOCKS].real for n in N], axis=1)
RIGHT = np.stack([np.diag(np.kron(IDENTITY, n.T))[BLOCKS].real for n in N], axis=1)


def superoperator(h, jumps):
    result = -1j * (np.kron(h, IDENTITY) - np.kron(IDENTITY, h.T))
    for a in jumps:
        w = a.conj().T @ a
        result += np.kron(a, a.conj()) - 0.5 * np.kron(w, IDENTITY) - 0.5 * np.kron(IDENTITY, w.T)
    return result[np.ix_(BLOCKS, BLOCKS)]


DISSIPATOR = superoperator(
    np.zeros((8, 8)), [np.sqrt(2 * GAMMA) * D[0].conj().T, np.sqrt(2 * GAMMA) * D[2]],
)


def unpack(vector):
    matrix = np.zeros(64, complex)
    matrix[BLOCKS] = vector
    return matrix.reshape(8, 8)


def gaussian(covariance):
    occupations, rotation = la.eigh((covariance + covariance.conj().T) / 2)
    rho = np.zeros((8, 8), complex)
    for bits in range(8):
        weight = 1.0
        state = IDENTITY[:, 0].copy()
        for k in range(3):
            if (bits >> k) & 1:
                creation = sum(rotation[i, k] * D[i].conj().T for i in range(3))
                state = creation @ state
                weight *= occupations[k]
            else:
                weight *= 1 - occupations[k]
        rho += weight * np.outer(state, state.conj())
    return rho


class ElectronDynamics:
    def __init__(self, h, rho, dt, nt):
        self.dt, self.nt = dt, nt
        self.weights = np.full(nt, dt)
        self.weights[[0, -1]] = dt / 2
        self.rho = rho.ravel()[BLOCKS]
        self.nobs = TRACE[:, None] * LEFT
        unitary = la.expm(-1j * h * dt)
        step = np.kron(unitary, unitary.conj())[np.ix_(BLOCKS, BLOCKS)]
        half = la.expm(DISSIPATOR * dt / 2)
        self.free = half @ step @ half
        loss = 1 - np.exp(-GAMMA * dt)
        self.count_R = loss * (self.nobs[:, 2] + self.nobs[:, 2] @ step @ half)
        injection = TRACE - self.nobs[:, 0]
        self.count_L = loss * (injection + injection @ step @ half)
        self.F = torch.tensor(self.free, dtype=CT)
        self.p0 = torch.tensor(self.rho, dtype=CT)
        self.tr = torch.tensor(TRACE, dtype=CT)
        self.diff = torch.tensor(LEFT - RIGHT, dtype=RT)
        self.sum = torch.tensor(LEFT + RIGHT, dtype=RT)
        self.w = torch.tensor(self.weights, dtype=RT)
        self.insert = torch.tensor(np.stack((LEFT[:, 2], RIGHT[:, 2])), dtype=CT)

    def physical(self, v, retarded=False, with_charge=False):
        r = self.rho.copy()
        history, steps, counts = [], [], []
        q_left, q_right = 0.0, 0.0
        for k in range(self.nt):
            kick = np.exp(-0.5j * self.weights[k] * ((LEFT - RIGHT) @ v[k]))
            r = kick * r
            history.append(r.copy())
            counts.append((q_left, q_right))
            if retarded and k < self.nt - 1:
                following = np.exp(-0.5j * self.weights[k + 1] * ((LEFT - RIGHT) @ v[k + 1]))
                steps.append(following[:, None] * self.free * kick[None, :])
            r = kick * r
            if k < self.nt - 1:
                if with_charge:
                    q_right += float((self.count_R @ r).real)
                    q_left += float((self.count_L @ r).real)
                r = self.free @ r
        history = np.array(history)
        density = (history @ self.nobs).real
        if not retarded:
            return (density, np.array(counts)) if with_charge else density
        response = np.zeros((self.nt, 3, self.nt, 3))
        variations = np.empty((20, 0), complex)
        for i in range(self.nt):
            if i:
                variations = steps[i - 1] @ variations
                response[i, :, :i, :] = (self.nobs.T @ variations).real.reshape(3, i, 3)
            insertion = -1j * (LEFT - RIGHT) * history[i, :, None] * self.weights[i]
            variations = np.column_stack((variations, insertion))
        return density, response

    def contour(self, v, q):
        """Normalized mean and covariance, with independent forward/backward sources."""
        potential = v @ self.diff.T.to(CT) + 0.5 * q @ self.sum.T.to(CT)
        kicks = torch.exp(-0.5j * self.w[:, None] * potential)
        states, steps = [], []
        r = self.p0
        for k in range(self.nt):
            r = kicks[k] * r
            states.append(r)
            if k < self.nt - 1:
                steps.append(kicks[k + 1, :, None] * self.F * kicks[k, None, :])
            r = kicks[k] * r
            if k < self.nt - 1:
                r = self.F @ r
        normalization = self.tr @ r
        backward = [None] * self.nt
        b = self.tr * kicks[-1]
        backward[-1] = b
        for k in range(self.nt - 2, -1, -1):
            b = b @ steps[k]
            backward[k] = b
        mean = torch.stack([
            backward[k][None, :] * self.insert @ states[k] for k in range(self.nt)
        ])
        blocks = []
        variations = torch.empty((20, 0), dtype=CT)
        for i in range(self.nt):
            if i:
                variations = steps[i - 1] @ variations
            variations = torch.cat((variations, (self.insert * states[i][None, :]).T), dim=1)
            pair = backward[i][None, :] * self.insert @ variations
            pad = torch.zeros((2, 2 * (self.nt - i - 1)), dtype=CT)
            blocks.append(torch.cat((pair, pad), dim=1))
        lower = torch.cat(blocks, dim=0)
        indices = torch.arange(2 * self.nt) // 2
        pair = torch.where(indices[:, None] >= indices[None, :], lower, lower.T)
        order = torch.cat((torch.arange(0, 2 * self.nt, 2), torch.arange(1, 2 * self.nt, 2)))
        mean = mean.flatten()[order] / normalization
        pair = pair[order][:, order] / normalization
        return -1j * (pair - mean[:, None] * mean[None, :]), mean


def inverse_ks(e, s, potential, guess=None):
    pe, ps = e.rho.copy(), s.rho.copy()
    vs = np.zeros_like(potential)
    for k in range(e.nt - 1):
        w = e.weights[k]
        pe_next = e.free @ (np.exp(-1j * w * ((LEFT - RIGHT) @ potential[k])) * pe)
        target = (pe_next @ e.nobs).real
        x = (guess[k, :2].copy() if guess is not None else
             vs[k - 1, :2].copy() if k else potential[k, :2].copy())
        for _ in range(80):
            kick = np.exp(-1j * w * ((LEFT[:, :2] - RIGHT[:, :2]) @ x))
            z = kick * ps
            propagated = s.free @ z
            residual = (propagated @ s.nobs).real[:2] - target[:2]
            if max(abs(residual)) < 1e-12:
                break
            derivative = s.free @ (-1j * w * (LEFT[:, :2] - RIGHT[:, :2]) * z[:, None])
            jacobian = (s.nobs[:, :2].T @ derivative).real
            if np.linalg.cond(jacobian) > 1e13:
                raise RuntimeError(f"Singular KS density inversion at step {k}")
            step = la.solve(jacobian, residual)
            damping = 1.0
            for _ in range(20):
                trial = x - damping * step
                trial_state = s.free @ (
                    np.exp(-1j * w * ((LEFT[:, :2] - RIGHT[:, :2]) @ trial)) * ps
                )
                trial_residual = (trial_state @ s.nobs).real[:2] - target[:2]
                if np.linalg.norm(trial_residual) < np.linalg.norm(residual):
                    break
                damping *= 0.5
            else:
                raise RuntimeError(f"KS density-inversion line search failed at step {k}")
            x = trial
        else:
            raise RuntimeError(f"KS density inversion did not converge at step {k}")
        vs[k, :2] = x
        pe, ps = pe_next, propagated
    vs[-1] = vs[-2]
    return vs


class PhotonAction:
    def __init__(self, model, g, omega):
        self.model = model
        self.keep = np.delete(np.arange(2 * model.nt), 2 * model.nt - 1)
        times = np.tile(np.arange(model.nt) * model.dt, 2)
        weights = np.r_[model.weights, -model.weights]
        rank = np.r_[np.arange(model.nt), np.arange(2 * model.nt - 1, model.nt - 1, -1)]
        tau = times[:, None] - times[None, :]
        sign = np.where(rank[:, None] >= rank[None, :], 1, -1)
        covariance = -1j * g * g * np.exp(-1j * omega * tau * sign - KAPPA * np.abs(tau))
        kernel = weights[:, None] * covariance * weights[None, :]
        reduced = kernel[np.ix_(self.keep, self.keep)].copy()
        # The two contour branches share one turning-point insertion.
        a, b = model.nt - 1, 2 * model.nt - 1
        reduced[a, :] += kernel[b, self.keep]
        reduced[:, a] += kernel[self.keep, b]
        reduced[a, a] += kernel[b, b]
        self.kernel = torch.tensor(reduced, dtype=CT)
        self.identity = torch.eye(2 * model.nt - 1, dtype=CT)

    def derivative(self, v, method):
        q = torch.zeros((self.model.nt, 3), dtype=RT, requires_grad=True)
        chi, mean = self.model.contour(torch.tensor(v, dtype=CT), q.to(CT))
        chi, mean = chi[self.keep][:, self.keep], mean[self.keep]
        product = self.kernel @ chi
        correlation = (0.5j * torch.trace(product) if method == "oep" else
                       -0.5j * torch.logdet(self.identity - product))
        action = (0.5 * mean @ (self.kernel @ mean) + correlation).real
        gradient = torch.autograd.grad(action, q)[0] / self.model.w[:, None]
        return gradient.detach().numpy()


def ks_state(epsilon2, detuning):
    h = np.array([[detuning, 1, 0.5], [1, epsilon2, 1], [0.5, 1, -detuning]], complex)
    decay = np.diag(np.exp(-np.array([GAMMA, 0, GAMMA]) * DT / 2))
    injection = np.diag([1 - np.exp(-GAMMA * DT), 0, 0])
    unitary = la.expm(-1j * h * DT)
    step = decay @ unitary @ decay
    source = decay @ unitary @ injection @ unitary.conj().T @ decay + injection
    return la.solve_discrete_lyapunov(step, source), h


def initial_system():
    h = np.array([[0, 1, 0.5], [1, EPSILON_2, 1], [0.5, 1, 0]], complex)
    he = (sum(h[i, j] * (D[i].conj().T @ D[j]) for i in range(3) for j in range(3))
          + U * (N[0] @ N[1] + N[1] @ N[2]) + U_TILDE * N[0] @ N[2])
    free = ElectronDynamics(he, IDENTITY / 8, DT, 3).free
    matrix = np.eye(20) - free
    rhs = np.zeros(20, complex)
    matrix[0], rhs[0] = TRACE, 1
    rho = unpack(la.solve(matrix, rhs))
    density = np.array([np.trace(n @ rho).real for n in N])
    roots = []
    for guess in [[EPSILON_2, 0], [EPSILON_2, -1], [1, 0], [3, 0.1], [3.5, 0.2], [-8, -0.5]]:
        fit = least_squares(
            lambda v: ks_state(v[0], v[1])[0].diagonal().real[:2] - density[:2], guess,
            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=300,
        )
        if np.max(abs(fit.fun)) < 1e-10 and all(np.linalg.norm(fit.x - r) > 1e-7 for r in roots):
            roots.append(fit.x)
    if not roots:
        raise RuntimeError("No density-matched stationary KS state")
    roots.sort(key=lambda x: (x[0] - EPSILON_2) ** 2 + 2 * x[1] ** 2)
    covariance, hs3 = ks_state(*roots[0])
    rhos = gaussian(covariance)
    hs = sum(hs3[i, j] * (D[i].conj().T @ D[j]) for i in range(3) for j in range(3))
    nt = int(round(DURATION / DT)) + 1
    return ElectronDynamics(he, rho, DT, nt), ElectronDynamics(hs, rhos, DT, nt)


def exact_charge(e, g, omega):
    p = PHOTON_CUTOFF + 1
    a = np.diag(np.sqrt(np.arange(1, p)), 1).astype(complex)
    identity = np.eye(p)
    h = omega * a.conj().T @ a
    generator = -1j * (np.kron(h, identity) - np.kron(identity, h.T))
    jump = np.sqrt(2 * KAPPA) * a
    w = jump.conj().T @ jump
    generator += (np.kron(jump, jump.conj()) - 0.5 * np.kron(w, identity)
                  - 0.5 * np.kron(identity, w.T))
    photon_step = la.expm(generator * e.dt)
    displacement = a + a.conj().T
    r = e.rho[:, None, None] * np.outer(identity[:, 0], identity[:, 0])[None, :, :]
    counts = []
    q_left, q_right = 0.0, 0.0
    for i in range(e.nt):
        counts.append((q_left, q_right))
        if i == e.nt - 1:
            break
        kick = la.expm(-1j * e.weights[i] * (g * displacement))
        for j in range(20):
            if LEFT[j, 2]:
                r[j] = kick @ r[j]
            if RIGHT[j, 2]:
                r[j] = r[j] @ kick.conj().T
        reduced = r.reshape(20, -1) @ identity.ravel()
        q_right += float((e.count_R @ reduced).real)
        q_left += float((e.count_L @ reduced).real)
        r = (e.free @ r.reshape(20, -1) @ photon_step.T).reshape(20, p, p)
    return np.array(counts)


def self_consistent_potential(e, s, g, omega, method, tolerance):
    potential = np.zeros((e.nt, 3))
    guess = None
    photon = PhotonAction(e if method == "irr" else s, g, omega)
    for _ in range(65):
        if method == "irr":
            source = potential
            _, response = e.physical(source, retarded=True)
        else:
            source = inverse_ks(e, s, potential, guess)
            guess = source
            _, response = s.physical(source, retarded=True)
        derivative = photon.derivative(source, method)
        matrix = response[1:, :2, :-1, :2].reshape(2 * (e.nt - 1), 2 * (e.nt - 1))
        rhs = derivative[1:, :2].ravel()
        candidate = np.zeros_like(potential)
        with warnings.catch_warnings():
            warnings.simplefilter("error", la.LinAlgWarning)
            candidate[:-1, :2] = la.solve(matrix, rhs).reshape(-1, 2)
        candidate[-1, :2] = candidate[-2, :2]
        if np.max(abs(candidate - potential)) < tolerance:
            return potential
        potential = 0.25 * potential + 0.75 * candidate
    raise RuntimeError(f"{method}: photon-potential iteration did not converge")


def calculate_case(e, s, g, omega, tolerance=5e-11):
    _, zero = e.physical(np.zeros((e.nt, 3)), with_charge=True)
    data = {"time": np.arange(e.nt) * e.dt, "charge_zero": zero,
            "charge_exact": exact_charge(e, g, omega)}
    for method in METHODS:
        potential = self_consistent_potential(e, s, g, omega, method, tolerance)
        _, data["charge_" + method] = e.physical(potential, with_charge=True)
    return data


def make_plots(out: Path, spectrum: list[dict], trajectory: list[dict]) -> None:
    for rows, xkey, xlabel, ylabel, filename in [
        (spectrum, "omega", r"Resonator frequency $\omega_0/t_h$", r"$\Delta Q_R(T)/e$", "tqd_spectrum"),
        (trajectory, "time", r"Time $t\,t_h$", r"$\Delta Q_R(t)/e$", "tqd_time"),
    ]:
        fig, ax = plt.subplots(figsize=(3.45, 2.15))
        for method in ("exact",) + METHODS:
            label = "Joint reference" if method == "exact" else LABELS[method]
            ax.plot([row[xkey] for row in rows], [row["delta_Q_R_over_e_" + method] for row in rows],
                    linestyle=LINESTYLES[method], label=label, linewidth=1.7)
        ax.axhline(0, linestyle="--", linewidth=0.5)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        handles, labels = ax.get_legend_handles_labels()
        fig.tight_layout(pad=0.5)
        fig.savefig(out / (filename + ".pdf"), bbox_inches="tight")
        plt.close(fig)
    fig = plt.figure(figsize=(7.1, 0.3))
    fig.legend(handles, labels, loc="center", ncol=4, frameon=False, fontsize=9, handlelength=2.5)
    fig.savefig(out / "tqd_legend.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/test_3"))
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    e, s = initial_system()
    spectrum, trajectory = [], []
    for omega in np.linspace(1.5, 4.5, 21):
        data = calculate_case(e, s, 0.01, float(omega))
        row = {"omega": float(omega)}
        for method in ("exact",) + METHODS:
            row["delta_Q_R_over_e_" + method] = float(
                data["charge_" + method][-1, 1] - data["charge_zero"][-1, 1]
            )
        spectrum.append(row)
        if abs(omega - 3.0) < 1e-10:
            for k, t in enumerate(data["time"]):
                time_row = {"time": float(t)}
                for method in ("exact",) + METHODS:
                    time_row["delta_Q_R_over_e_" + method] = float(
                        data["charge_" + method][k, 1] - data["charge_zero"][k, 1]
                    )
                trajectory.append(time_row)
        print(f"Fig. 3: omega_0={omega:g}", flush=True)
    write_csv(out / "spectrum.csv", spectrum)
    write_csv(out / "time_trace.csv", trajectory)
    make_plots(out, spectrum, trajectory)
    print(f"Fig. 3 complete: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
