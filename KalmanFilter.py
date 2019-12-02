from numpy import loadtxt, ones, zeros, linspace
from numpy.linalg import lstsq
from matplotlib.pyplot import figure, show, savefig
import matplotlib.pyplot as plt
import numpy

##########################################################################
##########--------------------CALIBRATION CODE------------------##########
#Function definitions
def model_eqn(v, k):
	"""model equation for fitting voltage reading of IR sensor to distance"""
	return k[1] / (v - k[0])


def lls_fit(r, v):
	"""linear least squares fit of sensor data to model equation"""
	N = len(r)
	A = ones((N, 2))
	A[:, 1] = 1.0 / r
	k, res, rank, s = lstsq(A, v)
	return k

#Loading of calibration data
data_calibration = loadtxt('calibration.csv', delimiter=',', skiprows=1)
#r = true range, v1, v2, v3, v4 = voltage readings from ir1, ir2, ir3 amd ir4 respectively
#s5 and s6 = distance measurements of sonar1 and sonar2
r = data_calibration[:, 2]
v1 = data_calibration[:, 4]
v2 = data_calibration[:, 5]
v3 = data_calibration[:, 6]
v4 = data_calibration[:, 7]
s5 = data_calibration[:, 8]
s6 = data_calibration[:, 9]

#Extract readings and measurements to each sensor's datasheet specification valid operating range
index_low_1 = numpy.where(r == next(x for x in r if x >= 0.15))[0][0] #IR1 low bound = 15cm
index_low_2 = numpy.where(r == next(x for x in r if x >= 0.04))[0][0] #IR2 low bound = 4cm
index_low_3 = numpy.where(r == next(x for x in r if x >= 0.1))[0][0] #IR3 low bound = 10cm
index_low_4 = numpy.where(r == next(x for x in r if x >= 1))[0][0] #IR4 low bound = 1m

index_high_1 = numpy.where(r == next(x for x in r if x >= 1.5))[0][0] #IR1 up bound = 1.5m
index_high_2 = numpy.where(r == next(x for x in r if x >= 0.3))[0][0] #IR2 up bound = 30cm
index_high_3 = numpy.where(r == next(x for x in r if x >= 0.8))[0][0] #IR3 up bound = 80cm
#IR4 up bound = 5m but outside of operating range of 3m so not specified

#linear least squares fit hyperbolic function to each IR sensors using their valid operating range readings
k1 = lls_fit(r[index_low_1:index_high_1], v1[index_low_1:index_high_1]) 
k2 = lls_fit(r[index_low_2:index_high_2], v2[index_low_2:index_high_2])
k3 = lls_fit(r[index_low_3:index_high_3], v3[index_low_3:index_high_3])
k4 = lls_fit(r[index_low_4:], v4[index_low_4:])

################################################################################
#####------- Kalman filter demonstration program to estimate range --------#####
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import time
import math

Q =  1 # higher Q: higher gain, faster response but lower accuracy, lover Q: better accuracy but time lag introduced
FILTER_VALUE = 0.3 #cut off value off comparing previous state to current state
VERY_BIG_NUMBER = 1000.0 #used to set weighting of unreliable measurement to be disregarded

def fusion(Z1, Z2, varZ1, varZ2, x_model=None):
    """sensor fusion to calculate overall best linear unbiased estimate (BLUE)"""
    if x_model:
        if (abs(Z2 - x_model) > FILTER_VALUE):
            varZ2 = VERY_BIG_NUMBER
        if (abs(Z1 - x_model) > FILTER_VALUE):
            varZ1 = VERY_BIG_NUMBER
    Z_hat = (Z1 * 1 / varZ1 + Z2 * 1 / varZ2) / (1 / varZ1 + 1 / varZ2)
    # print((1/varZ1)/(1/varZ1 + 1/varZ2), ' ', (1/varZ2)/(1/varZ1 + 1/varZ2))
    var_hat = 1 / (1 / varZ1 + 1 / varZ2)
    return Z_hat, var_hat


# Load data
filename = 'training2.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Initalise error arrays for each sensor
error_1 = [];
error_2 = [];
error_3 = [];
error_4 = [];
error_5 = [];
error_6 = [];
dataStdDev = ones(6)

# Split into columns
sensors = [None] * 6  
#index, time, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

# Time step
time = [0] + time

# Process and measurement noise variances
interval_default = 20

#use a bad initial guess of starting position and give it a high variance to indicate uncertainty
mean_X_post = 0
var_X_post = 1000 ** 2

plt.figure(figsize=(24, 8))

for i in range(len(time) - 1):

    dt = time[i + 1] - time[i]
    v = velocity_command[i]

    # Position process noise standard deviation
    std_W = Q * dt
    var_W = std_W ** 2

    # Calculate prior estimate of position and its variance ( using motion model )
    mean_X_prior = mean_X_post + v * dt
    var_X_prior = var_X_post + var_W

    # Calculate predict range through voltage measurement of IR sensors
    r_predict_1 = model_eqn(raw_ir1[i], k1)
    r_predict_2 = model_eqn(raw_ir2[i], k2)
    r_predict_3 = model_eqn(raw_ir3[i], k3)
    r_predict_4 = model_eqn(raw_ir4[i], k4)
    
    #calculate error between prediction and prior state
    error_1.append(abs(r_predict_1 - mean_X_prior))
    error_2.append(abs(r_predict_2 - mean_X_prior))
    error_3.append(abs(r_predict_3 - mean_X_prior))
    error_4.append(abs(r_predict_4 - mean_X_prior))
    error_5.append(abs(sonar1[i] - mean_X_prior))
    error_6.append(abs(sonar2[i] - mean_X_prior))

    # Calculates binned variance
    if len(error_1) < interval_default:
        interval = len(error_1)
    else:
        interval = interval_default
        
    #Calculate standard deviation of each sensor measurement for range base on error
    dataStdDev[0] = numpy.std((error_1[i - interval:i]), ddof=1)
    dataStdDev[1] = numpy.std((error_2[i - interval:i]), ddof=1)
    dataStdDev[2] = numpy.std((error_3[i - interval:i]), ddof=1)
    dataStdDev[3] = numpy.std((error_4[i - interval:i]), ddof=1)
    dataStdDev[4] = numpy.std((error_5[i - interval:i]), ddof=1)
    dataStdDev[5] = numpy.std((error_6[i - interval:i]), ddof=1)

    for j in range(len(dataStdDev)):
        if (math.isnan(dataStdDev[j])) or (dataStdDev[j] == 0):
            dataStdDev[j] = eval('error_' + str(j + 1))[i]
            
	#fusing of sensor measurements and based on standard deviation of each sensor , weight their reliability
    Z_hat, var_hat = fusion(sonar1[i], sonar2[i], numpy.square(dataStdDev[4]), numpy.square(dataStdDev[5]), mean_X_prior)
    Z_hat, var_hat = fusion(Z_hat, r_predict_1, var_hat, numpy.square(dataStdDev[0]), mean_X_prior)
    Z_hat, var_hat = fusion(Z_hat, r_predict_2, var_hat, numpy.square(dataStdDev[1]), mean_X_prior)
    Z_hat, var_hat = fusion(Z_hat, r_predict_3, var_hat, numpy.square(dataStdDev[2]), mean_X_prior)
    Z_hat, var_hat = fusion(Z_hat, r_predict_4, var_hat, numpy.square(dataStdDev[3]), mean_X_prior)

    # Estimate position from measurement ( using sensor model )
    x_infer = Z_hat

    # Calculate Kalman gain
    K = var_X_prior/(var_hat + var_X_prior)

    # Calculate posterior estimate of position and its variance
    mean_X_post = mean_X_prior + K * (x_infer - mean_X_prior)
    var_X_post = (1 - K) * var_X_prior

    plt.plot(time[i], mean_X_post, '.', alpha=0.2)
    plt.plot(time[i], range_[i], '.r') #can be used to plot true range for training1.csv, training2.csv and calibration.csv

plt.ylabel('range[m]')
plt.xlabel('time[s]')   
plt.ylim([0, 3.5])
plt.show()
