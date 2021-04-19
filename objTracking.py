'''
    File name         : objTracking.py
    Description       : Main file for object tracking
    Source            : https://github.com/RahmadSadli/2-D-Kalman-Filter/blob/master/objTracking.py
    Date created      : 20/02/2020
    Python Version    : 3.7
'''

import cv2
from Detector import detect
from KalmanFilter import KalmanFilter

def main():

    # Create opencv video capture object
    VideoCap = cv2.VideoCapture('video/randomball.avi')

    #Variable used to control the speed of reading the video
    ControlSpeedVar = 100  #Lowest: 1 - Highest:100

    HiSpeed = 100

    #Create KalmanFilter object KF
    #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

    KF = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)

    debugMode=1

    while(True):
        # Read frame
        ret, frame = VideoCap.read()

        # Detect object
        centers = detect(frame,debugMode)

        # If centroids are detected then track them
        if (len(centers) > 0):

            # Draw the detected circle
            cv2.circle(frame, (int(centers[0][0]), int(centers[0][1])), 10, (0, 191, 255), 2)

            # Predict
            (x, y) = KF.predict()
            # Draw a rectangle as the predicted object position
            startX = int(x - 15)
            startY = int(y - 15)
            endX = int(x + 15)
            endY = int(y + 15)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

            # Update
            (x1, y1) = KF.update(centers[0])

            startX1 = int(x1 - 15)
            startY1 = int(y1 - 15)
            endX1 = int(x1 + 15)
            endY1 = int(y1 + 15)
            # Draw a rectangle as the estimated object position
            cv2.rectangle(frame, (startX1, startY1), (endX1, endY1), (0, 0, 255), 2)

            start_est = int(x1 + 15)
            end_est = int(y1 + 10)
            start_pred = int(x + 15)
            end_pred = int(y)
            start_measured = int(centers[0][0] + 15)
            end_measured = int(centers[0][1] - 15)

            cv2.putText(frame, "Estimated Position", (start_est, end_est), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Predicted Position", (start_pred, end_pred), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Measured Position", (start_measured, end_measured), 0, 0.5, (0,191,255), 2)

        cv2.imshow('image', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(HiSpeed-ControlSpeedVar+1)


if __name__ == "__main__":
    # execute main
    main()
