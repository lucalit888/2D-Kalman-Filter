'''
    Author        : Luca Lit
    Description   : KalmanFilter class used for object tracking
    Source        : https://github.com/RahmadSadli/2-D-Kalman-Filter/blob/master/objTracking.py
    Last edited   : 06/08/2021
'''

import numpy as np

class KalmanFilter(object):
    '''
    Parameter descriptions:
        delta_t: time interval
        acc_x: acceleration in x-direction
        acc_y: acceleration in y-direction
        std_acc: process noise magnitude
        x_std_meas: standard deviation of the measurement in x-direction
        y_std_meas: standard deviation of the measurement in y-direction
    '''

    def __init__(self, delta_t, acc_x, acc_y, std_acc, x_std_meas, y_std_meas):
        self.delta_t = delta_t
        self.x_aprev = np.matrix([[acc_x],[acc_y]])
        self.x_kprev = np.matrix([[0], [0], [0], [0]])

        ''' 
                      A                              B
        -----------------------------------------------------------------          
                [ 1  0  Δt  0 ]              [ 0.5(Δt)^2  0 ]
        x_k =   [ 0  1  0  Δt ] . x_{k-1} +  [ 0  0.5(Δt)^2 ] . a_{k-1}
                [ 0  0  1   0 ]              [ Δt         0 ]
                [ 0  0  0   1 ]              [ 0          Δt]
        
        where x_k is the current state (position) of the object, x_k-1 is the previous state/position, and a_k-1 is the previous acceleration in x,y directions
        '''

        # Defining State Transition Matrix A
        self.A = np.matrix([[1, 0, self.delta_t, 0],
                            [0, 1, 0, self.delta_t],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Define Control Input Matrix B
        self.B = np.matrix([[(self.delta_t**2)/2, 0],
                            [0, (self.delta_t**2)/2],
                            [self.delta_t, 0],
                            [0, self.delta_t]])

        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Initial Process Noise Covariance
        self.Q = np.matrix([[(self.delta_t**4)/4, 0, (self.delta_t**3)/2, 0],
                            [0, (self.delta_t**4)/4, 0, (self.delta_t**3)/2],
                            [(self.delta_t**3)/2, 0, self.delta_t**2, 0],
                            [0, (self.delta_t**3)/2, 0, self.delta_t**2]]) * std_acc**2

        # Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2,0],
                           [0, y_std_meas**2]])

        # Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])

    # Prediction method that returns the predicted state position (x,y) of the object
    def predict(self):
        # Update time state
        #  x_k = A * x_(k-1) + B * a_(k-1)     Eq.(9)
        self.x_kprev = np.dot(self.A, self.x_kprev) + np.dot(self.B, self.x_aprev)

        # Calculate error covariance
        # P= A*P*A' + Q                        Eq.(10)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x_kprev[0:2]

    # Prediction that is adjusted using Kalman Filter, returns (x,y)
    def KF_adjust(self, z):
        # S = H* P * H' + R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H * P * H' + R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x_kprev = np.round(self.x_kprev + np.dot(K, (z - np.dot(self.H, self.x_kprev))))

        '''
        I = [[1. 0. 0. 0.]
            [0. 1. 0. 0.]
            [0. 0. 1. 0.]
            [0. 0. 0. 1.]]
        '''
        I = np.eye(self.H.shape[1])
        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P   #Eq.(13)
        return self.x_kprev[0:2]
