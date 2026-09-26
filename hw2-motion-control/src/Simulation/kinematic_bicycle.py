import numpy as np
import sys
sys.path.append("..")
from Simulation.utils import State, ControlState
from Simulation.kinematic import KinematicModel

class KinematicModelBicycle(KinematicModel):
    def __init__(self,
            l = 30,     # distance between rear and front wheel
            dt = 0.05
        ):
        # Distance from center to wheel
        self.l = l
        # Simulation delta time
        self.dt = dt

    def step(self, state:State, cstate:ControlState) -> State:
            # TODO 2.3.1: Bicycle Kinematic Model
        a = cstate.a
        delta = cstate.delta

        x = state.x
        y = state.y
        yaw = state.yaw
        v = state.v

        yaw_rad = np.deg2rad(yaw)
        delta_rad = np.deg2rad(delta)

        # use current state to update pose
        x = x + v * np.cos(yaw_rad) * self.dt
        y = y + v * np.sin(yaw_rad) * self.dt

        # bicycle yaw rate
        w_rad = v / self.l * np.tan(delta_rad)
        yaw = (yaw + np.rad2deg(w_rad * self.dt)) % 360

        # update velocity
        v = v + a * self.dt

        # store angular velocity in degree/s
        w = np.rad2deg(w_rad)
        # [end] TODO 2.3.1

        state_next = State(x, y, yaw, v, w)
        return state_next
