import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerStanleyBicycle(Controller):
    def __init__(self, model, 
                 # TODO 4.3.1: Tune Stanley Gain
                 kp=0.5):
        self.path = None
        self.kp = kp
        self.l = model.l
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.current_idx = 0

    # State: [x, y, yaw, delta, v]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, delta, v = info["x"], info["y"], info["yaw"], info["delta"], info["v"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 5:
            return 0.0

        # Search Front Wheel Target Locally
        front_x = x + self.l*np.cos(np.deg2rad(yaw))
        front_y = y + self.l*np.sin(np.deg2rad(yaw))
        vf = v / np.cos(np.deg2rad(delta)) if np.cos(np.deg2rad(delta)) != 0 else v
        
        min_idx, min_dist = utils.search_nearest_local(self.path, (front_x,front_y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        target = self.path[min_idx]
        # TODO 4.3.1: Stanley Control for Bicycle Kinematic Model
        path_yaw = np.deg2rad(target[2])

        # heading error
        heading_error = path_yaw - np.deg2rad(yaw)
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # signed cross-track error measured at front wheel
        dx = front_x - target[0]
        dy = front_y - target[1]
        cte = -dx * np.sin(path_yaw) + dy * np.cos(path_yaw)

        next_delta = heading_error + np.arctan2(-self.kp * cte, vf + 1e-6)
        next_delta = np.rad2deg(next_delta)
        # [end] TODO 4.3.1
    
        return next_delta
