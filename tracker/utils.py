import numpy as np

class KalmanFilter(object):
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = np.zeros((dim_x, 1))       # state
        self.z = np.zeros((dim_z, 1))       # measurement
        self.F = np.eye(dim_x)              # system(transition) matrix
        self.H = np.zeros((dim_z, dim_x))   # measurement matrix
        self.P = np.eye(dim_x)              # state estimation error matrix
        self.Q = np.eye(dim_x)              # process noise matrix
        self.R = np.eye(dim_z)              # measurement noise matrix
        self.K = np.zeros((dim_x, dim_z))   # kalman gain

    # time update
    def predict(self):
        F = self.F
        Q = self.Q

        # x = Fx + Bu
        # ignore control input
        self.x = np.dot(F, self.x)

        # P = FPF' + Q
        self.P = np.dot(np.dot(F, self.P), F.T) + Q
    
    # measurement update
    def update(self, z):
        if z.shape != (self.dim_z, 1):
            raise ValueError("z must be (%d, 1) shape"%(self.dim_z))

        # compute kalman gain
        P = self.P
        H = self.H
        R = self.R

        PHT = np.dot(P, H.T)
        # K = PH'inv(HPH' + R)
        K = np.dot(PHT, np.linalg.inv(np.dot(H, PHT) + R))
        self.K = K

        # update estimate with measurement z
        x = self.x
        # x = x + K(z - Hx)
        self.x = x + np.dot(K, (z - np.dot(H, x)))

        # update the error matrix
        # P = (I - KH)P + KRK'
        self.P = np.dot((np.eye(self.dim_x) - np.dot(K, H)), P) + np.dot(np.dot(K, R), K.T)