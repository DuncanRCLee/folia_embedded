import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R

DefaultConfig = {}
DefaultConfig = {
"sigma_a": sigma_a,
"sigma_w": sigma_w,
"g": 9.8029,
"T": T,
"dt": dt,
    }

DefaultConfig["var_a"] = np.power(self.sigma_a,2)
self.var_w = np.power(self.sigma_w,2)
self.config["var_w"] = self.var_w
self.g = self.config["g"]
self.T = self.config["T"]
##process noise in body frame
self.sigma_acc = 0.5*np.ones((1,3))
self.var_acc = np.power(self.sigma_acc,2)
self.sigma_gyro = 0.5*np.ones((1,3))*np.pi/180
self.var_gyro = np.power(self.sigma_gyro,2)

self.Q = np.zeros((6,6))  ##process noise covariance matrix Q
self.Q[0:3,0:3] = self.var_acc*np.identity(3)
self.Q[3:6,3:6] = self.var_gyro*np.identity(3)
self.config["Q"] = self.Q

self.sigma_vel = 0.01 #0.01 default
self.R = np.zeros((3,3))
self.R[0:3,0:3] = np.power(self.sigma_vel,2)*np.identity(3)   ##measurement noise, 0.01 default
self.config["R"] = self.R

self.H = np.zeros((3,9))
self.H[0:3,3:6] = np.identity(3)
self.config["H"]= self.H      

class LiveLocalizer:
    def __init__(self, config):
        self.config = config
        self.imu_buffer = []
        self.x = None
        self.q = None
        self.P_hat = None
        self.initialized = False

    def initialize(self, imu0_window):
        imu0_window = np.asarray(imu0_window)
        x = np.zeros(9)
        # Estimate initial attitude from accelerometer
        avg_acc = imu0_window[:, 0:3].mean(axis=0)
        roll = np.arctan2(-avg_acc[1], -avg_acc[2])
        pitch = np.arctan2(avg_acc[0], np.linalg.norm(avg_acc[1:3]))
        heading = 0.0

        x[6:9] = [roll, pitch, heading]
        # Create initial quaternion (SciPy uses [x, y, z, w])
        rot0 = R.from_euler('xyz', [roll, pitch, heading], degrees=False)
        q = rot0.as_quat()

        # Initial covariance
        P_hat = np.zeros((9, 9))
        P_hat[0:3, 0:3] = (1e-5)**2 * np.eye(3)
        P_hat[3:6, 3:6] = (1e-5)**2 * np.eye(3)
        P_hat[6:9, 6:9] = (0.1 * np.pi / 180)**2 * np.eye(3)

        self.x = x
        self.q = q
        self.P_hat = P_hat
        self.initialized = True

    def step(self, imu_sample, dt):
        assert self.P_hat is not None, "should not be none" 

        self.imu_buffer.append(imu_sample)

        self.x, self.q, Rot = self.nav_eq(self.x, imu_sample, self.q, dt)
        F, G = self.state_update(imu_sample, self.q, dt)
        Q = self.config["Q"]

        self.P_hat = F @ self.P_hat @ F.T + G @ Q @ G.T

        # ZUPT detection
        if self.detect_zupt(self.imu_buffer[-5:]):
            self.x, self.P_hat, self.q = self.corrector(self.x, self.P_hat, Rot)

        return self.x

    def nav_eq(self, x_in, imu, q_in, dt):
        x_out = x_in.copy()
        # Current orientation
        rot = R.from_quat(q_in)
        # Integrate angular rate: delta rotation
        omega = imu[3:6]
        delta_rot = R.from_rotvec(omega * dt)
        rot_out = rot * delta_rot
        q_out = rot_out.as_quat()

        # Update attitude in state
        x_out[6:9] = rot_out.as_euler('xyz', degrees=False)

        # Transform acceleration to nav frame
        Rot_mat = rot_out.as_matrix()
        acc_n = Rot_mat @ imu[0:3] + np.array([0, 0, self.config["g"]])
        # Update velocity and position
        x_out[3:6] += dt * acc_n
        x_out[0:3] += dt * x_out[3:6] + 0.5 * dt**2 * acc_n

        return x_out, q_out, Rot_mat

    def state_update(self, imu, q, dt):
        F = np.eye(9)
        F[0:3, 3:6] = dt * np.eye(3)

        Rot_mat = R.from_quat(q).as_matrix()
        imu_r = Rot_mat @ imu[0:3]
        f_skew = np.array([
            [0, -imu_r[2], imu_r[1]],
            [imu_r[2], 0, -imu_r[0]],
            [-imu_r[1], imu_r[0], 0]
        ])
        F[3:6, 6:9] = -dt * f_skew

        G = np.zeros((9, 6))
        G[3:6, 0:3] = dt * Rot_mat
        G[6:9, 3:6] = -dt * Rot_mat

        return F, G

    def corrector(self, x_check, P_check, Rot_mat):
        H = self.config["H"]
        R_cov = self.config["R"]
        K = P_check @ H.T @ LA.inv(H @ P_check @ H.T + R_cov)
        z = -x_check[3:6]

        dx = K @ z
        x_check += dx

        # Correct attitude with small-angle approximation
        omega = dx[6:9]
        omega_skew = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
        Rot_corr = (np.eye(3) + omega_skew) @ Rot_mat
        rot = R.from_matrix(Rot_corr)
        q_new = rot.as_quat()
        x_check[6:9] = rot.as_euler('xyz', degrees=False)

        # Symmetrize covariance
        P_check = (np.eye(9) - K @ H) @ P_check
        P_check = 0.5 * (P_check + P_check.T)

        return x_check, P_check, q_new

    def detect_zupt(self, recent_window, threshold=3e8):
        if len(recent_window) < 5:
            return False

        acc = np.array([s[0:3] for s in recent_window])
        gyro = np.array([s[3:6] for s in recent_window])
        mean_acc = acc.mean(axis=0)
        score = 0.0
        for a, w in zip(acc, gyro):
            acc_term = ((a - self.config["g"] * mean_acc / LA.norm(mean_acc)) @
                        (a - self.config["g"] * mean_acc / LA.norm(mean_acc)).T)
            gyro_term = w @ w.T
            score += acc_term / self.config["var_a"] + gyro_term / self.config["var_w"]
        score /= len(recent_window)
        return score < threshold
