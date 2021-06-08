'''
    File name         : objTracking.py
    Name              : Luca Lit
    Description       : Main file for object tracking (currently deals with constant velocity)
    Adapted from      : https://github.com/RahmadSadli/2-D-Kalman-Filter/blob/master/objTracking.py
    Date created      : 03/05/2021
    Python Version    : 3.7
    To-Do             : plotting covariance
'''

from KalmanFilter import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import math
from object import Object

# create a new class

# PL compute the distance between two points as euclidean space

# Generates array of (x,y) pairs representing motion over time
# Parameters:
# total_time: total time elapsed
# omega: radians per second
# acceleration: meters per second^2
def generate_motion(total_time, velocity, theta, omega, acceleration, delta_t, start_pos_x, start_pos_y):
    i = 0
    positions_over_time = []
    x = start_pos_x; y = start_pos_y;
    while i < total_time:
        theta = theta + (omega * delta_t)
        velocity = velocity + (acceleration * delta_t)
        x = x + (velocity * math.cos(theta))
        y = y + (velocity * math.sin(theta))
        positions_over_time.append(np.array([[x], [y]]))
        i += delta_t

    return positions_over_time


def main():
    KF_obsA = KalmanFilter(1, 1, 1, 1, 0.1, 0.1)
    KF_obsB = KalmanFilter(1, 1, 1, 1, 0.1, 0.1)
    ax = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    ObstacleA = Object(2, math.pi/4, 2*math.pi/180, 0.5, 5, 5)
    ASV = Object(2, math.pi/4, -2*math.pi/180, 1, 2, 80)

    obstacle_positions_over_time= []
    ASV_positions_over_time = []
    total_time = 20
    delta_t = 1
    i = 0
    while i < total_time:
        # Move both ObstacleA and ASV objects
        ObstacleA.move(delta_t); ASV.move(delta_t)
        # Track their new positions
        obstacle_newpos = ObstacleA.position(); ASV_newpos = ASV.position()
        print(obstacle_newpos)
        ax.plot(obstacle_newpos[0], obstacle_newpos[1], "o", markerfacecolor="green", alpha=0.5, markeredgecolor='None',
                label="obstacle observed value")
        ax.plot(ASV_newpos[0], ASV_newpos[1], "o", markerfacecolor="red", alpha=0.5, markeredgecolor='None',
                label="ASV position")
        ax2.plot(ASV_newpos[0], ASV_newpos[1], "o", markerfacecolor="red", alpha=0.5, markeredgecolor='None',
                label="ASV position")

        obstacle_positions_over_time.append(np.array([[obstacle_newpos[0]], [obstacle_newpos[1]]]))
        ASV_positions_over_time.append(np.array([[ASV_newpos[0]], [ASV_newpos[1]]]))

        ############## Converting from global to local frame --> P_G to P_L first ##############
        # Defining the transformation matrix that takes into account rotation + movement
        T = np.array([
            [math.cos(ASV.gettheta()), -1 * math.sin(ASV.gettheta()), 0, ASV_newpos[0]],
            [math.sin(ASV.gettheta()), math.cos(ASV.gettheta()), 0, ASV_newpos[1]],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        # print(T)
        T_inv = np.linalg.inv(T)
        # print(T_inv)

        P_L = np.dot(T_inv, np.array([obstacle_newpos[0], obstacle_newpos[1], 0, 1]))

        ############## Converting from local to global frame --> P_L to P_G ##############
        P_G = np.dot(T, P_L)
        # print("original", np.array([obstacle_newpos[0], obstacle_newpos[1], 0, 1]))
        # print("converted P_G", P_G)

        ############# Running prediction ##############
        (x, y) = KF_obsA.predict()
        ax2.plot(x, y, "o", markerfacecolor="None", markeredgecolor='orange', label = "predicted")

        ############## Running Kalman Filter to update centroid ##############
        (x1, y1) = KF_obsA.update(np.array([[P_G[0]], [P_G[1]]]))
        print("(x1,y1)", (x1, y1))
        ax2.plot(x1, y1, "o", markerfacecolor="None", markeredgecolor='blue', label = "kalman")

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.pause(0.6)
        i += delta_t

    plt.show()


if __name__ == "__main__":
    # execute main
    main()
