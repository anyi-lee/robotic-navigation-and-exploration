import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPurePursuitBicycle(Controller):
    def __init__(self, model, 
                 # TODO 4.3.1: Tune Pure Pursuit Gain
                 kp=0.2, Lfc=10.0):
        self.path = None
        self.kp = kp
        self.Lfc = Lfc
        self.dt = model.dt
        self.l = model.l
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.current_idx = 0

    # State: [x, y, yaw, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, v = info["x"], info["y"], info["yaw"], info["v"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 5:
            return 0.0

        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        
        Ld = self.kp*v + self.Lfc
        
        # TODO 4.3.1: Pure Pursuit Control for Bicycle Kinematic Model
        target_idx = min_idx
        for i in range(min_idx, len(self.path)):
            dx = self.path[i][0] - x
            dy = self.path[i][1] - y
            if np.hypot(dx, dy) >= Ld:
                target_idx = i
                break

        target = self.path[target_idx]
        tx, ty = target[0], target[1]

        # heading to look-ahead point
        alpha = np.arctan2(ty - y, tx - x) - np.deg2rad(yaw)
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        Ld = max(Ld, 1e-6)

        # clamp steering to avoid unstable oversteer
        next_delta = np.rad2deg(np.arctan2(2.0 * self.l * np.sin(alpha), Ld))
        next_delta = np.clip(next_delta, -30.0, 30.0)

        # [end] TODO 4.3.1
        return next_delta
