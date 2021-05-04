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

# Generates array of (x,y) pairs representing motion over time
def generate_motion(total_time, delta_t, start_pos_x, start_pos_y):
    i = 0
    positions_over_time = []
    x = start_pos_x; y = start_pos_y;
    while i < total_time:
        x = x + (i ** 2)/5
        y = y + (i ** 2)/5
        positions_over_time.append(np.array([[x], [y]]))
        i += delta_t

    return positions_over_time


def main():

    KF = KalmanFilter(0.5, 1, 1, 1, 0.1, 0.1)

    ax = plt.subplot(1, 1, 1)

    motion_positions = generate_motion(20, 0.5, 0, 0)

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
            (x1, y1) = KF.update(centers)
            ax.plot(x1, y1, "o", markerfacecolor="None", markeredgecolor='blue', label = "kalman")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.pause(0.6)

    plt.show()

if __name__ == "__main__":
    # execute main
    main()
