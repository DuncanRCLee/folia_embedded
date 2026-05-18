# pyviz/filter3/mathutil.py
import jax
import jax.numpy as jnp
import equinox as eqx

# Utility small constant based on dtype
def _eps_for(x):
    x = jnp.asarray(x)
    dtype = x.dtype
    # use a modest epsilon relative to floating point precision
    return jnp.array(1e-12, dtype=dtype)


def quat_normalize(q, eps=None):
    """
    Normalize quaternion(s) `q` in a numerically stable, differentiable way.

    Accepts arrays of shape (..., 4) and returns same shape.
    """
    q = jnp.asarray(q)
    if eps is None:
        eps = _eps_for(q)
    norm2 = jnp.sum(q * q, axis=-1, keepdims=True)
    # Use sqrt(max(norm2, eps)) to avoid NaNs/inf in gradients for zero input
    denom = jnp.sqrt(jnp.maximum(norm2, eps))
    return q / denom


def quat_mul(q1, q2):
    """
    Hamilton (quaternion) product q = q1 * q2.

    Supports broadcasting; inputs should have last dimension of size 4.
    Returns an array with the broadcasted leading dims and last dim 4.
    Uses only JAX-friendly ops and avoids python-level control flow that would
    break vectorization or differentiation.
    """
    q1 = jnp.asarray(q1)
    q2 = jnp.asarray(q2)

    # Ensure trailing dimension exists
    if q1.shape[-1] != 4 or q2.shape[-1] != 4:
        raise ValueError("Quaternions must have last dimension 4")

    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    w2 = q2[..., 0]
    x2 = q2[..., 1]
    y2 = q2[..., 2]
    z2 = q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return jnp.stack([w, x, y, z], axis=-1)


def quat_from_small(dtheta):
    """
    Convert a small rotation vector (axis-angle with small angle) `dtheta` (shape ... x 3)
    into a quaternion (... x 4) using the exponential map:
      q = [cos(|θ|/2), (θ/|θ|) * sin(|θ|/2)]
    For small |θ| this uses a Taylor series expansion to avoid division-by-zero and
    preserve stable gradients.

    This is differentiable for all inputs.
    """
    dtheta = jnp.asarray(dtheta)
    if dtheta.shape[-1] != 3:
        raise ValueError("dtheta must have last dimension 3")

    theta = jnp.linalg.norm(dtheta, axis=-1)  # shape (...)
    dtype = dtheta.dtype
    eps = _eps_for(dtheta)

    half = 0.5 * theta  # (...)
    w = jnp.cos(half)  # (...)

    # Compute scaling factor s = sin(half) / theta robustly.
    # For theta -> 0, sin(theta/2)/theta -> 1/2 - theta^2/48 + O(theta^4)
    # Use a Taylor expansion for small theta to avoid dividing by zero.
    # We compute a second-order Taylor approximation.
    small = theta < 1e-3  # threshold is JAX-trace-friendly boolean array
    # safe division for non-small cases
    s_non_small = jnp.sin(half) / jnp.where(theta > eps, theta, eps)
    # Taylor approx: sin(theta/2)/theta ≈ 1/2 - theta^2 / 48
    s_taylor = 0.5 - (theta * theta) / 48.0
    s = jnp.where(small, s_taylor, s_non_small)  # shape (...)

    # Expand s to vector shape
    s_expanded = jnp.expand_dims(s, axis=-1)  # (..., 1)
    vec = dtheta * s_expanded  # (..., 3)

    q = jnp.concatenate([jnp.expand_dims(w, axis=-1), vec], axis=-1)  # (...,4)

    # Return normalized quaternion to reduce error accumulation
    return quat_normalize(q)


def quat_to_R(q):
    """
    Convert quaternion(s) `q` (... x 4) to rotation matrix(es) (... x 3 x 3).
    The quaternion is normalized in a stable differentiable way before conversion.

    Formula uses standard quaternion-to-rotation-matrix components.
    """
    q = jnp.asarray(q)
    if q.shape[-1] != 4:
        raise ValueError("Quaternion must have last dimension 4")

    q = quat_normalize(q)
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    # Precompute repeated terms
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    # Each entry has shape (...)
    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)

    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)

    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    # Stack into (..., 3, 3)
    row0 = jnp.stack([m00, m01, m02], axis=-1)
    row1 = jnp.stack([m10, m11, m12], axis=-1)
    row2 = jnp.stack([m20, m21, m22], axis=-1)

    mat = jnp.stack([row0, row1, row2], axis=-2)  # (..., 3, 3)
    return mat