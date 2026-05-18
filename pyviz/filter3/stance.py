import jax
import jax.numpy as jnp
import equinox as eqx
from mathutil import quat_mul, quat_from_small, quat_normalize, quat_to_R


class DiffStanceDetector(eqx.Module):
    """Differentiable stance detection module (JAX + Equinox).

    This module computes a smooth stance probability from IMU data
    using differentiable heuristics inspired by Zero-Velocity Update
    (ZUPT) methods. It combines multiple soft thresholds:
      1. Accelerometer magnitude near g
      2. Low variance in acceleration
      3. Low gyro magnitude
      4. Small integrated Euler distance
    """


    #sigmoid thresholds
    th_acc_mag: jnp.ndarray   # threshold for |a|-g
    th_acc_var: jnp.ndarray   # threshold for accel variance
    th_gyro_mag: jnp.ndarray  # threshold for gyro magnitude
    th_euler: jnp.ndarray     # threshold for integrated displacement

    k_acc_mag: jnp.ndarray    # sigmoid sharpness for |a|-g
    k_acc_var: jnp.ndarray
    k_gyro_mag: jnp.ndarray
    k_euler: jnp.ndarray

    g: float = eqx.static_field()
    win_var: int = eqx.static_field()
    win_euler: int = eqx.static_field()

    # ------------------------------------------------------------------
    def __init__(self,
                 th_acc_mag=0.5,
                 th_acc_var=0.05,
                 th_gyro_mag=0.5,
                 th_euler=0.005,
                 k=40.0,
                 g=9.80665,
                 win_var=10,
                 win_euler=20):
        # trainable params
        self.th_acc_mag = jnp.array(th_acc_mag)
        self.th_acc_var = jnp.array(th_acc_var)
        self.th_gyro_mag = jnp.array(th_gyro_mag)
        self.th_euler = jnp.array(th_euler)

        self.k_acc_mag = jnp.array(k)
        self.k_acc_var = jnp.array(k)
        self.k_gyro_mag = jnp.array(k)
        self.k_euler = jnp.array(k)

        # frozen params
        self.g = float(g)
        self.win_var = int(win_var)
        self.win_euler = int(win_euler)

    # ------------------------------------------------------------------
    def _rolling_var(self, x, W):
        """Rolling variance via E[x^2] - (E[x])^2; differentiable."""
        pad = jnp.repeat(x[:1], W - 1)
        xpad = jnp.concatenate([pad, x], axis=0)
        csum = jnp.cumsum(xpad)
        csum2 = jnp.cumsum(xpad**2)
        wsum = csum[W:] - csum[:-W]
        wsum2 = csum2[W:] - csum2[:-W]
        mean = wsum / W
        mean2 = wsum2 / W
        return jnp.maximum(0.0, mean2 - mean**2)

    # ------------------------------------------------------------------
    def _euler_distance_window(self, acc_win, gyro_win, dt):
        """Integrate motion over a window to compute displacement norm."""
        def step(carry, inputs):
            p, v, q = carry
            a_k, w_k = inputs
            q = quat_mul(q, quat_from_small(w_k * dt))
            R = quat_to_R(q)
            a_world = R @ a_k + jnp.array([0.0, 0.0, -self.g])
            v = v + a_world * dt
            p = p + v * dt
            return (p, v, q), None

        p0 = jnp.zeros(3)
        v0 = jnp.zeros(3)
        q0 = jnp.array([1., 0., 0., 0.])
        (pT, _, _), _ = jax.lax.scan(step, (p0, v0, q0), (acc_win, gyro_win))
        return jnp.linalg.norm(pT)

    # ------------------------------------------------------------------
    def __call__(self, acc, gyro, dt):
        """Compute stance probability for each timestep."""
        N = acc.shape[0]
        g = self.g

        # (1) "near-g" criterion
        a_mag = jnp.linalg.norm(acc, axis=1)
        s_acc_mag = jax.nn.sigmoid(self.k_acc_mag * (self.th_acc_mag - jnp.abs(a_mag - g)))

        # (2) low-variance criterion
        var_a = self._rolling_var(a_mag, self.win_var)
        s_acc_var = jax.nn.sigmoid(self.k_acc_var * (self.th_acc_var - var_a))

        # (3) low-gyro criterion
        gyro_mag = jnp.linalg.norm(gyro, axis=1)
        s_gyro = jax.nn.sigmoid(self.k_gyro_mag * (self.th_gyro_mag - gyro_mag))

        # (4) small integrated motion (Euler distance)
        W = self.win_euler
        acc_pad = jnp.concatenate([jnp.repeat(acc[:1], W - 1, axis=0), acc], axis=0)
        gyro_pad = jnp.concatenate([jnp.repeat(gyro[:1], W - 1, axis=0), gyro], axis=0)

        def euler_at(i):
            acc_w = jax.lax.dynamic_slice_in_dim(acc_pad, i, W, axis=0)
            gyro_w = jax.lax.dynamic_slice_in_dim(gyro_pad, i, W, axis=0)
            d = self._euler_distance_window(acc_w, gyro_w, dt)
            return jax.nn.sigmoid(self.k_euler * (self.th_euler - d))

        s_euler = jax.vmap(euler_at)(jnp.arange(N))

        # Combine all criteria multiplicatively
        stance_prob = s_acc_mag * s_acc_var * s_gyro * s_euler
        return stance_prob
