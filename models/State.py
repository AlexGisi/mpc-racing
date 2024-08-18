from dataclasses import dataclass


@dataclass
class State:
    x: float  # Global X position
    y: float  # Global Y position
    yaw: float  # Angle with the X axis
    v_x: float  # Velocity in the x direction in the local frame
    v_y: float  # Velocity in the y direciton in the local frame
    yaw_dot: float

    # Nice to include commands for e.g. passing initial values
    # to MPC
    steer: float = None
    throttle: float = None

    def set_controls(self, throttle, steer):
        self.throttle = throttle
        self.steer = steer
        return self

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, State):
            return False

        return (
            self.x == value.x
            and self.y == value.y
            and self.yaw == value.yaw
            and self.v_x == value.v_x
            and self.v_y == value.v_y
            and self.yaw_dot == value.yaw_dot
            and self.steer == value.steer
            and self.throttle == value.throttle
        )
