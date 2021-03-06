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
        print(x,y)
        positions_over_time.append(np.array([[x], [y]]))
        i += delta_t

    return positions_over_time


def main():
    # ObstacleA = Object(2, math.pi/4, 2*math.pi/180, 0.5, 5, 5)
    # ASV = Object(2, math.pi/4, 2*math.pi/180, 1, 0, 0)
    #
    # total_time = 20
    # delta_t = 1
    # i = 0
    # while i < total_time:
    #
    #     i += delta_t

    KF = KalmanFilter(1, 1, 1, 1, 0.1, 0.1)

    ax = plt.subplot(1, 1, 1)

    # (2, math.pi / 4, 0.2 * math.pi / 180, 0.5, 5, 5)
    motion_positions = generate_motion(20, 2, math.pi/4, 2*math.pi/180, 0.5, 1, 5, 5)

    # Find max/min of x,y for axis limits
    x_coords = []
    y_coords = []
    for i in range(len(motion_positions)):
        x_coords.append(motion_positions[i][0][0])
        y_coords.append(motion_positions[i][1][0])

    plt.xlim([0, max(x_coords) + 10])
    plt.ylim([0, max(y_coords) + 10])

    for i in range(len(motion_positions)):

        # Detect object
        centers = motion_positions[i]
        # print(centers)

        # If centroids are detected then track them
        if (len(centers) > 0):

            x_coords.append(centers[0])
            y_coords.append(centers[1])

            # Draw the detected circle
            ax.plot(centers[0], centers[1], "o", markerfacecolor="black", alpha=0.5, markeredgecolor='None', label = "observed value")

            # Predict
            (x, y) = KF.predict()
            # Draw a rectangle as the predicted object position
            ax.plot(x, y, "o", markerfacecolor="None", markeredgecolor='orange', label = "predicted")

            # Update
            print(centers)
            (x1, y1) = KF.update(centers)
            print("(x1,y1)", (x1, y1))
            ax.plot(x1, y1, "o", markerfacecolor="None", markeredgecolor='blue', label = "kalman")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.pause(0.6)

    plt.show()

if __name__ == "__main__":
    # execute main
    main()
