# object class
import numpy as np
import math

class Object:
    def __init__(self, velocity, theta, omega, acceleration, start_pos_x, start_pos_y):
       self.velocity = velocity
       self.theta = theta
       self.omega = omega
       self.acceleration = acceleration
       self.x = start_pos_x
       self.y = start_pos_y

    def move(self, delta_t):
        self.theta = self.theta + (self.omega * delta_t)
        self.velocity = self.velocity + (self.acceleration * delta_t)
        self.x = self.x + (self.velocity * math.cos(self.theta))
        self.y = self.y + (self.velocity * math.sin(self.theta))

    def position(self):
        return (self.x, self.y)

    def gettheta(self):
        return self.theta

    # def generate_motion(total_time, velocity, theta, omega, acceleration, delta_t, start_pos_x, start_pos_y):
    #     i = 0
    #     positions_over_time = []
    #     x = start_pos_x; y = start_pos_y;
    #     while i < total_time:
    #         theta = theta + (omega * delta_t)
    #         velocity = velocity + (acceleration * delta_t)
    #         x = x + (velocity * math.cos(theta))
    #         y = y + (velocity * math.sin(theta))
    #         positions_over_time.append(np.array([[x], [y]]))
    #         i += delta_t
    #
    #     return positions_over_time