import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerLQRBicycle(Controller):
    def __init__(self, model, Q=None, R=None, control_state='steering_angle'):
        self.path = None
        if control_state == 'steering_angle':
            self.Q = np.eye(2)
            self.R = np.eye(1)
            # TODO 4.4.1: Tune LQR Gains
            self.Q[0,0] = 8.0
            self.Q[1,1] = 3.0
            self.R[0,0] = 1.0
        elif control_state == 'steering_angular_velocity':
            self.Q = np.eye(3)
            self.R = np.eye(1)
            # TODO 4.4.4: Tune LQR Gains
            self.Q[0,0] = 8.0
            self.Q[1,1] = 3.0
            self.Q[2,2] = 1.0
            self.R[0,0] = 1.0
        self.pe = 0
        self.pth_e = 0
        self.pdelta = 0
        self.dt = model.dt
        self.l = model.l
        self.control_state = control_state
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.pe = 0
        self.pth_e = 0
        self.pdelta = 0
        self.current_idx = 0

    def _solve_DARE(self, A, B, Q, R, max_iter=150, eps=0.01): # Discrete-time Algebra Riccati Equation (DARE)
        P = Q.copy()
        for i in range(max_iter):
            temp = np.linalg.inv(R + B.T @ P @ B)
            Pn = A.T @ P @ A - A.T @ P @ B @ temp @ B.T @ P @ A + Q
            if np.abs(Pn - P).max() < eps:
                break
            P = Pn
        return Pn

    # State: [x, y, yaw, delta, v]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, delta, v = info["x"], info["y"], info["yaw"], info["delta"], info["v"]
        yaw = utils.angle_norm(yaw)

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 3:
            return 0.0
        
        # Search Nesrest Target
        min_idx, min_dist = utils.search_nearest(self.path, (x,y))
        target = self.path[min_idx]
        target[2] = utils.angle_norm(target[2])
        
        if self.control_state == 'steering_angle':
            # TODO 4.4.1: LQR Control for Bicycle Kinematic Model with steering angle as control input
            path_yaw = np.deg2rad(target[2])
            yaw_rad = np.deg2rad(yaw)

            heading_error = path_yaw - yaw_rad
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

            dx = x - target[0]
            dy = y - target[1]
            e = -dx * np.sin(path_yaw) + dy * np.cos(path_yaw)

            kappa = target[3]
            delta_ff = np.arctan(self.l * kappa)

            A = np.array([
                [1.0, -v * self.dt],
                [0.0,  1.0]
            ])
            B = np.array([
                [0.0],
                [-v * self.dt / self.l]
            ])

            X = np.array([[e], [heading_error]])
            P = self._solve_DARE(A, B, self.Q, self.R)
            K = np.linalg.inv(B.T @ P @ B + self.R) @ (B.T @ P @ A)

            u = -(K @ X)[0, 0]
            next_delta = np.rad2deg(delta_ff + u)
            # [end] TODO 4.4.1
        elif self.control_state == 'steering_angular_velocity':
            # TODO 4.4.4: LQR Control for Bicycle Kinematic Model with steering angular velocity as control input
            path_yaw = np.deg2rad(target[2])
            yaw_rad = np.deg2rad(yaw)
            delta_rad = np.deg2rad(delta)

            heading_error = path_yaw - yaw_rad
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

            dx = x - target[0]
            dy = y - target[1]
            e = -dx * np.sin(path_yaw) + dy * np.cos(path_yaw)

            kappa = target[3]
            delta_ff = np.arctan(self.l * kappa)
            delta_err = delta_rad - delta_ff

            A = np.array([
                [1.0, -v * self.dt, 0.0],
                [0.0,  1.0,        -v * self.dt / self.l],
                [0.0,  0.0,         1.0]
            ])
            B = np.array([
                [0.0],
                [0.0],
                [self.dt]
            ])

            X = np.array([[e], [heading_error], [delta_err]])
            P = self._solve_DARE(A, B, self.Q, self.R)
            K = np.linalg.inv(B.T @ P @ B + self.R) @ (B.T @ P @ A)

            delta_dot = -(K @ X)[0, 0]
            next_delta = np.rad2deg(delta_rad + delta_dot * self.dt)
            # [end] TODO 4.4.4
        
        return next_delta
