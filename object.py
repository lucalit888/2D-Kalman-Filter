'''
    Author        : Luca Lit
    File Purpose  : Defining the Object Class
    Last edited   : 06/08/2021
    Description   : Creating a class which represents an obstacle / object
                    Parameters of the object include: velocity, angle of rotation, change in rotation over time, acceleration, start pos X and Y
'''

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

    # function which moves the object by certain velocity and rotation in one timeframe delta_t
    def move(self, delta_t):
        self.theta = self.theta + (self.omega * delta_t)
        self.velocity = self.velocity + (self.acceleration * delta_t)
        self.x = self.x + (self.velocity * math.cos(self.theta))
        self.y = self.y + (self.velocity * math.sin(self.theta))

    # returns position of object
    def position(self):
        return (self.x, self.y)

    # returns the rotation of the object
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