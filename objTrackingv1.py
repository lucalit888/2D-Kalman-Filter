'''
    File name         : objTracking.py
    Name              : Luca Lit
    Description       : Main file for object tracking (currently deals with constant velocity)
    Adapted from      : https://github.com/RahmadSadli/2-D-Kalman-Filter/blob/master/objTracking.py
    Date created      : 03/05/2021
    Python Version    : 3.7
'''

import cv2
from Detector import detect
from KalmanFilter import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

def main():


    KF = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)

    n = 0
    m = 0
    positions_over_time = [np.array([[n], [m]])]

    x_coords = []
    y_coords = []
    ax = plt.subplot(1, 1, 1)
    plt.xlim([0, 250])
    plt.ylim([0, 250])

    num_of_points = 25

    for i in range(num_of_points):
        n = n + 10
        # m = m + (n*n/100)
        m = m + 5
        positions_over_time.append(np.array([[n], [m]]))

    for i in range(num_of_points):

        # Detect object
        centers = positions_over_time[i]

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


if __name__ == "__main__":
    # execute main
    main()
