"""
ESKF Functional Core (JAX)
--------------------------
- Nominal state x = {p, q, v, ba, bg}
- Error state δx = [δp, δθ, δv, δba, δbg]
- Prediction: strapdown INS with biases; covariance from error dynamics
- Update: generic linearized measurement (e.g., ZUPT: z=0, h(x)=v)
- Reset: inject δx into x, then covariance adjustment G P G^T for rotation

This module is **functional** (no side-effects), JIT/scan-ready, and differentiable.
You can plug any tuning method (grad-based, EM/Bayesian, manual) around it.

Assumes JAX is available. All arrays are float64 for stability.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any

# ----------------------------
# Utilities: quaternions and SO(3)
# ----------------------------

def skew(v: jnp.ndarray) -> jnp.ndarray:
    x, y, z = v
    return jnp.array([[0.0, -z,  y],
                      [z,  0.0, -x],
                      [-y, x,  0.0]], dtype=v.dtype)


def quat_mul(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product. q = [qw, qx, qy, qz]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=q1.dtype)


def quat_normalize(q: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    return q / jnp.clip(jnp.linalg.norm(q), eps, jnp.inf)


def quat_from_small_angle(dtheta: jnp.ndarray) -> jnp.ndarray:
    """First-order: q{δθ} = [1, δθ/2]."""
    return jnp.concatenate([jnp.array([1.0], dtheta.dtype), 0.5*dtheta])


def quat_from_omega_dt(omega_dt: jnp.ndarray) -> jnp.ndarray:
    """Rotation from body angular increment (rad) using small-angle for tiny dt"""
    angle = jnp.linalg.norm(omega_dt)
    half = 0.5 * angle
    def small():
        return jnp.concatenate([jnp.array([1.0], omega_dt.dtype), 0.5*omega_dt])
    def large():
        axis = omega_dt / (angle + 1e-15)
        return jnp.concatenate([jnp.array([jnp.cos(half)], omega_dt.dtype), axis*jnp.sin(half)])
    return lax.cond(angle < 1e-6, small, large)


def quat_to_R(q: jnp.ndarray) -> jnp.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q
    R = jnp.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=q.dtype)
    return R

# ----------------------------
# State container (PyTree)
# ----------------------------
@dataclass
class Nominal:
    p: jnp.ndarray  # (3,)
    q: jnp.ndarray  # (4,) unit quaternion [w,x,y,z]
    v: jnp.ndarray  # (3,)
    ba: jnp.ndarray # (3,)
    bg: jnp.ndarray # (3,)

    def as_vector(self) -> jnp.ndarray:
        return jnp.concatenate([self.p, self.q, self.v, self.ba, self.bg])

    @staticmethod
    def from_vector(x: jnp.ndarray) -> "Nominal":
        return Nominal(p=x[0:3], q=x[3:7], v=x[7:10], ba=x[10:13], bg=x[13:16])

STATE_DIM = 15  # error-state dim: 3+3+3+3+3

# ----------------------------
# Prediction (strapdown INS)
# ----------------------------
# Gravity constant used for nominal propagation.
GRAVITY_WORLD = jnp.array([0.0, 0.0, -9.81])


def predict_nominal(x: Nominal, am: jnp.ndarray, wm: jnp.ndarray, dt: float) -> Nominal:
    """Nominal propagation with bias-compensated IMU."""
    Rwb = quat_to_R(x.q)
    a_nav = Rwb @ (am - x.ba) + GRAVITY_WORLD
    p = x.p + x.v*dt + 0.5*a_nav*(dt*dt)
    v = x.v + a_nav*dt
    dq = quat_from_omega_dt((wm - x.bg)*dt)
    q = quat_normalize(quat_mul(x.q, dq))
    # Biases random-walk in error dynamics; nominal stay constant
    return Nominal(p=p, q=q, v=v, ba=x.ba, bg=x.bg)

# ----------------------------
# Error-state linearization F, G (continuous time)
# ----------------------------
@dataclass
class Lin:
    F: jnp.ndarray  # (15,15)
    G: jnp.ndarray  # (15,12)  mapping process noises to δx dot


def linearize_error(x: Nominal, am: jnp.ndarray, wm: jnp.ndarray) -> Lin:
    Rwb = quat_to_R(x.q)
    # Notation: process noises: an (accel noise), wn (gyro noise), aw (accel bias RW), ww (gyro bias RW)
    Z3 = jnp.zeros((3,3), x.p.dtype)
    I3 = jnp.eye(3, dtype=x.p.dtype)

    a_tilde = am - x.ba
    omega_tilde = wm - x.bg

    F = jnp.zeros((STATE_DIM, STATE_DIM), x.p.dtype)
    # δp dot = δv
    F = F.at[0:3, 6:9].set(I3)
    # δv dot = -R*(a_tilde)_x δθ - R δba - R an
    F = F.at[6:9, 3:6].set(-Rwb @ skew(a_tilde))
    F = F.at[6:9, 9:12].set(-Rwb)
    # δθ dot = - (ω_tilde)_x δθ - δbg - wn
    F = F.at[3:6, 3:6].set(-skew(omega_tilde))
    F = F.at[3:6, 12:15].set(-I3)
    # δba dot = aw; δbg dot = ww  (in F, zeros; driven via G)

    G = jnp.zeros((STATE_DIM, 12), x.p.dtype)
    # an -> δv
    G = G.at[6:9, 0:3].set(-Rwb)
    # wn -> δθ
    G = G.at[3:6, 3:6].set(-I3)
    # aw -> δba
    G = G.at[9:12, 6:9].set(I3)
    # ww -> δbg
    G = G.at[12:15, 9:12].set(I3)

    return Lin(F=F, G=G)


def discretize(F: jnp.ndarray, G: jnp.ndarray, Qc: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """First-order (Euler) discretization: Φ ≈ I+F dt, Qd ≈ G Qc G^T dt.
    For higher accuracy, replace with Van Loan or midpoint.
    """
    I = jnp.eye(F.shape[0], F.dtype)
    Phi = I + F*dt
    Qd = G @ Qc @ G.T * dt
    # Symmetrize for numerical stability
    Qd = 0.5 * (Qd + Qd.T)
    return Phi, Qd

# ----------------------------
# Update (generic), with Cholesky solves
# ----------------------------
@dataclass
class UpdateOut:
    x_upd: Nominal
    P_upd: jnp.ndarray
    innov: jnp.ndarray
    S: jnp.ndarray
    K: jnp.ndarray


def chol_solve(L: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    y = jax.scipy.linalg.solve_triangular(L, B, lower=True)
    return jax.scipy.linalg.solve_triangular(L.T, y, lower=False)


def ekf_update(x_pred: Nominal,
               P_pred: jnp.ndarray,
               H: jnp.ndarray,
               z: jnp.ndarray,
               h_of_x: jnp.ndarray,
               R: jnp.ndarray) -> UpdateOut:
    # Innovation and its covariance
    y = z - h_of_x  # residual
    S = H @ P_pred @ H.T + R
    S = 0.5 * (S + S.T) + 1e-12*jnp.eye(S.shape[0], S.dtype)
    L = jnp.linalg.cholesky(S)
    K = (P_pred @ H.T) @ chol_solve(L, jnp.eye(S.shape[0], S.dtype))

    # Error-state correction
    dx = K @ y
    # Inject error into nominal state
    x_upd = inject_error(x_pred, dx)

    I = jnp.eye(P_pred.shape[0], P_pred.dtype)
    P_upd = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T
    P_upd = 0.5 * (P_upd + P_upd.T)

    # Reset (covariance adjustment for rotation error)
    G_reset = reset_jacobian(dx)
    P_upd = G_reset @ P_upd @ G_reset.T

    return UpdateOut(x_upd=x_upd, P_upd=P_upd, innov=y, S=S, K=K)

# ----------------------------
# Error injection and reset
# ----------------------------

def inject_error(x: Nominal, dx: jnp.ndarray) -> Nominal:
    dp = dx[0:3]
    dtheta = dx[3:6]
    dv = dx[6:9]
    dba = dx[9:12]
    dbg = dx[12:15]

    p = x.p + dp
    q = quat_normalize(quat_mul(x.q, quat_from_small_angle(dtheta)))
    v = x.v + dv
    ba = x.ba + dba
    bg = x.bg + dbg
    return Nominal(p=p, q=q, v=v, ba=ba, bg=bg)


def reset_jacobian(dx: jnp.ndarray) -> jnp.ndarray:
    """G ≈ diag(I6, I3 - [1/2 δθ]_x, I9).
    Implements the standard ESKF covariance reset after injecting rotation error.
    """
    dtheta = dx[3:6]
    I3 = jnp.eye(3, dtheta.dtype)
    Z3 = jnp.zeros((3,3), dtheta.dtype)
    G = jnp.eye(STATE_DIM, dtheta.dtype)
    G = G.at[3:6, 3:6].set(I3)
    G = G.at[6:9, 6:9].set(I3)
    G = G.at[0:3, 0:3].set(I3)
    # Rotation block adjustment:
    G_theta = I3 - 0.5 * skew(dtheta)
    G = G.at[3:6, 3:6].set(G_theta)
    return G

# ----------------------------
# One full ESKF step (predict covariance + optional update)
# ----------------------------
@dataclass
class StepParams:
    Qc: jnp.ndarray       # (12,12) continuous-time noise for [an, wn, aw, ww]
    R_meas: jnp.ndarray   # (m,m) measurement noise for current update (if used)

    def tree_flatten(self):
        return (self.Qc, self.R_meas), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        Qc, R_meas = children
        return cls(Qc=Qc, R_meas=R_meas)


def eskf_step(carry: Tuple[Nominal, jnp.ndarray, StepParams],
              inputs: Dict[str, Any]) -> Tuple[Tuple[Nominal, jnp.ndarray, StepParams], Dict[str, Any]]:
    x, P, sp = carry
    am = inputs["am"]  # (3,)
    wm = inputs["wm"]  # (3,)
    dt = inputs["dt"]  # scalar

    # Nominal predict
    x_pred = predict_nominal(x, am, wm, dt)

    # Covariance predict
    lin = linearize_error(x, am, wm)
    Phi, Qd = discretize(lin.F, lin.G, sp.Qc, dt)
    P_pred = Phi @ P @ Phi.T + Qd
    P_pred = 0.5 * (P_pred + P_pred.T)

    # Optional update (ZUPT example)
    do_update = inputs.get("do_update", False)

    def do_upd_fun(_):
        # Example: ZUPT -> z = 0, h(x) = v, H selects δv
        H = jnp.zeros((3, STATE_DIM), P_pred.dtype)
        H = H.at[:, 6:9].set(jnp.eye(3, P_pred.dtype))
        z = jnp.zeros(3, P_pred.dtype)
        h_of_x = x_pred.v
        out = ekf_update(x_pred, P_pred, H, z, h_of_x, sp.R_meas)
        return out.x_upd, out.P_upd, out, True

    def skip_upd_fun(_):
        dummy = UpdateOut(x_upd=x_pred, P_upd=P_pred, innov=jnp.zeros(3), S=jnp.eye(3), K=jnp.zeros((STATE_DIM,3)))
        return x_pred, P_pred, dummy, False

    x_new, P_new, upd_out, used = lax.cond(do_update, do_upd_fun, skip_upd_fun, operand=None)

    outputs = {
        "x": x_new,
        "P": P_new,
        "innov": upd_out.innov,
        "used_update": used
    }
    return (x_new, P_new, sp), outputs

# ----------------------------
# Rollout over a sequence
# ----------------------------

def eskf_rollout(x0: Nominal,
                 P0: jnp.ndarray,
                 Qc: jnp.ndarray,
                 R_meas: jnp.ndarray,
                 am_seq: jnp.ndarray,  # (T,3)
                 wm_seq: jnp.ndarray,  # (T,3)
                 dt_seq: jnp.ndarray,  # (T,)
                 zupt_flags: jnp.ndarray | None = None):
    T = am_seq.shape[0]
    if zupt_flags is None:
        zupt_flags = jnp.zeros((T,), dtype=bool)

    sp = StepParams(Qc=Qc, R_meas=R_meas)

    def _step(carry, t):
        inputs = {
            "am": am_seq[t],
            "wm": wm_seq[t],
            "dt": dt_seq[t],
            "do_update": zupt_flags[t]
        }
        return eskf_step(carry, inputs)

    (xT, PT, _), traj = lax.scan(_step, (x0, P0, sp), jnp.arange(T))
    return traj  # dict of arrays PyTree: {x, P, innov, used_update}

# ----------------------------
# Example loss hooks (for tuning)
# ----------------------------

def mse_position_loss(est_positions: jnp.ndarray, gt_positions: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((est_positions - gt_positions) ** 2)


def innovations_mse(innov_seq: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(innov_seq ** 2)

# You can write a training loop that calls eskf_rollout, then computes a loss
# from traj["x"].p stacked over time, and backprop into parameters like Qc, R, or even bias models.
