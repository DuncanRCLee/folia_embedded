from typing import NamedTuple
import jax
import jax.numpy as jnp
import jax.lax as lax
import equinox as eqx

class NominalState(NamedTuple):
    p: jnp.ndarray # (3,)
    v: jnp.ndarray # (3,)
    q: jnp.ndarray # (4,)
    a_b: jnp.ndarray # (3,)
    w_b: jnp.ndarray # (3,)


class ESKF(eqx.Module):
    F: eqx.Module # the state transition function
    h: eqx.Module # the observation function
    Q: jnp.ndarray # ()
    R: jnp.ndarray # (4,4)

    detector: eqx.Module # the observation detector


    def predict():
        pass

    def update():
        pass


    def inject_error():
        pass

    def full_step():
        pass




# math
def skew(v):
    x,y,z = v
    return jnp.array([[0,-z, y],[z,0,-x],[-y,x,0]], v.dtype)

def quat_mul(q1,q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], q1.dtype)

def quat_normalize(q, eps=1e-12):
    return q / jnp.clip(jnp.linalg.norm(q), eps, jnp.inf)

def quat_from_small_angle(dtheta):
    return jnp.concatenate([jnp.array([1.0], dtheta.dtype), 0.5*dtheta])

def quat_from_omega_dt(omega_dt):
    angle = jnp.linalg.norm(omega_dt)
    half = 0.5*angle
    def small():
        return jnp.concatenate([jnp.array([1.0], omega_dt.dtype), 0.5*omega_dt])
    def large():
        axis = omega_dt / (angle + 1e-15)
        return jnp.concatenate([jnp.array([jnp.cos(half)], omega_dt.dtype), axis*jnp.sin(half)])
    return lax.cond(angle < 1e-6, small, large)

def quat_to_R(q):
    q = quat_normalize(q)
    w,x,y,z = q
    return jnp.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], q.dtype)
