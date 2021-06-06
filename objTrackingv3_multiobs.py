'''
    File name         : objTracking.py
    Name              : Luca Lit
    Description       : Main file for object tracking (currently deals with constant velocity)
    Adapted from      : https://github.com/RahmadSadli/2-D-Kalman-Filter/blob/master/objTracking.py
    Date created      : 03/05/2021
    Python Version    : 3.7
    To-Do             : plotting covariance
'''

import cv2
from Detector import detect
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
    plt.xlim([-100, 260])
    plt.ylim([0, 450])
    ax2 = plt.subplot(1, 2, 2)
    plt.xlim([-100, 260])
    plt.ylim([0, 450])

    ObstacleA = Object(2, math.pi/4, 2*math.pi/180, 0.2, 5, 5)
    ObstacleB = Object(2, -math.pi/4, math.pi/180, -0.5, 200, 100)
    Obstacles = [ObstacleA, ObstacleB]
    ASV = Object(2, math.pi/4, -2*math.pi/180, 1, 2, 80)

    obstacle_positions_over_time= []
    ASV_positions_over_time = []
    total_time = 20
    delta_t = 1
    i = 0
    while i < total_time:
        # Move both ObstacleA and ASV objects
        for obstacle in Obstacles:
            obstacle.move(delta_t)
            obstacle.move(delta_t)
        ASV.move(delta_t)
        # Track their new positions
        obstacleA_newpos = ObstacleA.position()
        obstacleB_newpos = ObstacleB.position()
        ASV_newpos = ASV.position()

        ax.plot(obstacleA_newpos[0], obstacleA_newpos[1], "o", markerfacecolor="green", alpha=0.5, markeredgecolor='None',
                label="obstacle A actual")
        ax.plot(obstacleB_newpos[0], obstacleB_newpos[1], "o", markerfacecolor="purple", alpha=0.5, markeredgecolor='None',
                label="obstacle B actual")
        ax.plot(ASV_newpos[0], ASV_newpos[1], "o", markerfacecolor="red", alpha=0.5, markeredgecolor='None',
                label="ASV position")
        ax2.plot(ASV_newpos[0], ASV_newpos[1], "o", markerfacecolor="red", alpha=0.5, markeredgecolor='None',
                label="ASV position")

        # obstacle_positions_over_time.append(np.array([[obstacle_newpos[0]], [obstacle_newpos[1]]]))
        # ASV_positions_over_time.append(np.array([[ASV_newpos[0]], [ASV_newpos[1]]]))

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

        P_L_A = np.dot(T_inv, np.array([obstacleA_newpos[0], obstacleA_newpos[1], 0, 1]))
        P_L_B = np.dot(T_inv, np.array([obstacleB_newpos[0], obstacleB_newpos[1], 0, 1]))

        ############## Converting from local to global frame --> P_L to P_G ##############
        P_G_A = np.dot(T, P_L_A)
        P_G_B = np.dot(T, P_L_B)

        # print("original", np.array([obstacle_newpos[0], obstacle_newpos[1], 0, 1]))
        # print("converted P_G", P_G)

        ############# Running prediction ##############
        (x_A, y_A) = KF_obsA.predict()
        ax2.plot(x_A, y_A, "o", markerfacecolor="None", markeredgecolor='brown', label = "predicted Obs.A")

        (x_B, y_B) = KF_obsB.predict()
        ax2.plot(x_B, y_B, "o", markerfacecolor="None", markeredgecolor='orange', label = "predicted Obs.B")

        ############## Running Kalman Filter to update centroid ##############
        (x1_A, y1_A) = KF_obsA.update(np.array([[P_G_A[0]], [P_G_A[1]]]))
        (x1_B, y1_B) = KF_obsB.update(np.array([[P_G_B[0]], [P_G_B[1]]]))

        ax2.plot(x1_A, y1_A, "o", markerfacecolor="None", markeredgecolor='blue', label = "Kalman Obs.A")
        ax2.plot(x1_B, y1_B, "o", markerfacecolor="None", markeredgecolor='teal', label = "Kalman Obs.B")

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.pause(0.6)
        i += delta_t

    plt.show()


if __name__ == "__main__":
    # execute main
    main()
