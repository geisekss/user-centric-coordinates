%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
import matplotlib.mlab as mlab
from sklearn.neighbors import KernelDensity
from scipy.stats import pearsonr
import math
from scipy.signal import *


eps = 0.000000001

'''
    The spherical linear interpolation implementation to interpolate the quaternion data
'''
def slerp3(rot):
    idxs = np.array([3,0,1,2])
    nrot = np.copy(rot)
    for i in range(1,rot.shape[0]-1):
        ql = mathutils.Quaternion(np.copy(rot[i-1,idxs]))
        qr = mathutils.Quaternion(np.copy(rot[i+1,idxs]))
        qc = mathutils.Quaternion(np.copy(rot[i,idxs]))

        q1 = ql.slerp(qr, 0.5)
        nc = q1.slerp(qc, 0.5)
        nrot[i, idxs] = np.array(nc)
    return nrot


'''
    Some low-pass filter implementations: moving average filter, butterworth filter, and box-data filter
'''
def movingAverage_filter(acc, w):
    weigths = w/sum(w)
    m = int(len(w)/2)
    nacc = [[np.dot(acc[k-m:k+m+1, i], weigths) for i in range(acc.shape[1])] for k in range(m, acc.shape[0]-m)]
    return np.array(nacc)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    y = lfilter(b, a, data)
    return y       
               
      
def box_data_filter(data, w, idxs):
    new_data = np.copy(data)    
    
    weights_vector = np.ones(w)    
    weights = np.array(weights_vector)/float(np.sum(weights_vector))  
    
    len_before = int((weights.shape[0]-1)/2)
    len_after = len(weights) - len_before - 1
    pos_start = len_before
    pos_final = len(new_data) - len_after - 1
    
    for i in idxs:
        new_data[pos_start:pos_final+1, i] = np.convolve(data[:, i], weights, 'valid')
        
    for pos in range(0, pos_start):
        for i in idxs:
            new_data[pos,i] = conv_border(data[:pos+len_after,i], len_after+pos)
        
    for pos in range(data.shape[0]-1, pos_final, -1):
        for i in idxs:
            new_data[pos,i] = conv_border(data[pos-len_before+1:,i], len_before+(data.shape[0]-1-pos))
    return new_data



def zero_pole_filter(f, fs):
    w = (2*math.pi*f)/fs
    r = 0.9
    z0 = [math.cos(w), math.sin(w)]
    zp = [r*math.cos(w), r*math.sin(w)]
    b = [1, -2*z0[0], (z0[0]*z0[0])+(z0[1]*z0[1])]
    a = [1, -2*zp[0], (zp[0]*zp[0])+(zp[1]*zp[1])]
    return b, a


'''
    Apply a bandstop filter of a given frequency in a temporal vector acc collected in a sampling frequency Fs. The filter is applied in each axis singly
'''
def notch_filter(acc, fr=False):
    nacc = np.copy(acc)
    b, a =  zero_pole_filter(0, Fs)
    if(fr): freq_response_filter(b, a)
    for i in range(acc.shape[1]):
        nacc[:, i] = lfilter(b, a, nacc[:, i])
    return nacc


def low_pass_filter(acc, fr=False):
    nacc = np.copy(acc)
    b = firwin(numtaps=16, cutoff=10, nyq=Fs/2.0)
    if(fr): freq_response_filter(b)
    for i in range(acc.shape[1]):
        nacc[:, i] = lfilter(b, 1, nacc[:, i])
    return nacc


'''
    The velocities are obtained given a n-dimensional temporal vector of n triaxial acceleration measurements
'''
def calculate_velocities(real_world):
    v_samples = np.zeros(real_world.shape, dtype=float)
    # divide by 10ˆ3 when the timestamp is in nanoseconds, and remove the 10ˆ3 when is milliseconds
    for i in range(1,real_world.shape[0]):
        for idx_c in range(3):
            v_samples[i, idx_c] = v_samples[i-1, idx_c] + (period/1000.0) * real_world[i-1, idx_c] + (period/2000.0)*(real_world[i, idx_c]- real_world[i-1, idx_c])
    return v_samples



def getRealWorldCoordinates(acc, rotation_vector):
    # Android rotation vector is the axis [x,y,z] first and then the angle w. This transforms the rotation vector sample in a quaternion form
    w, x1b, y1b, z1b = rotation_vector[3], rotation_vector[0], rotation_vector[1], rotation_vector[2]

    x1, y1, z1 = -x1b, -y1b, -z1b
    
    x2, y2, z2 = acc        
        
    x_w = math.pow(w,2)*x2 + math.pow(x1,2)*x2 - x2*(math.pow(y1,2) + math.pow(z1,2)) + 2*w*(y2*z1 - y1*z2) + 2*x1*(y1*y2 + z1*z2)
    y_w = math.pow(w,2)*y2 - math.pow(x1,2)*y2 + math.pow(y1,2)*y2 - 2*w*x2*z1 - y2*math.pow(z1,2) + 2*y1*z1*z2 + 2*x1*(x2*y1 + w*z2)
    z_w = 2*w*x2*y1 - 2*w*x1*y2 + 2*x1*x2*z1 + 2*y1*y2*z1 + math.pow(w,2)*z2 - math.pow(x1,2)*z2 - math.pow(y1,2)*z2 + math.pow(z1,2)*z2
        
    return [x_w, y_w, z_w]


 '''
    Calculate the world coordinates given an acceleration vector with n measurements and 
    the corresponding rotation vector with n measurements
'''
def calculateRealWorldCoordinates(acc_data, rot_data):
    real_world_coords = np.zeros(acc_data.shape, dtype=float)
    
    for i in range(acc_data.shape[0]):
        real_world_coords[i] = getRealWorldCoordinates(acc_data[i], rot_data[i])
                
    return  real_world_coords

'''
    Get the user coordinates for n acceleration measurements acc in world coordinates given its velocity values
'''
def calculateUserCoordinates(real_world, v_samples):
    samples_rotated = np.copy(real_world)
    for i in range(1, samples_rotated.shape[0]):
        if(np.linalg.norm(v_samples[i, :]) < eps):
            v_samples[i, :] = v_samples[i-1, :]
        e_1 = v_samples[i, :]/np.linalg.norm(v_samples[i, :])
        e_2 = np.cross(e_1, calculate_gravity(real_world, i))
        e_2 = e_2/np.linalg.norm(e_2)
        R_u_w = np.vstack((e_1[:2], e_2[:2])).T
        samples_rotated[i, :2] = np.dot(R_u_w, real_world[i, :2])     
    return v_samples, samples_rotated


if __name__ == "__main__":
    '''
        Given the interpolated accelerometer and rotation vector data, and the sampling rate, calculate the real world coordinates and the user-centric coordinates
    '''
    Fs = float(sys.argv[1])
    period = (1/Fs) * 1000
    acc_interp_file = argv[2]
    rot_interp_file = argv[3]
    acc = np.loadtxt(acc_interp_file, delimiter=',')
    rot = np.loadtxt(rot_interp_file, delimiter=',')
    real_world = calculateRealWorldCoordinates(acc, rot)
    real_world = movingAverage_filter(real_world, 9)
    real_world = notch_filter(real_world)
    velocities = calculate_velocities(real_world)
    velocities, user_coord = calculateUserCoordinates(real_world, velocities)
    user_coord = notch_filter(user_coord)
    user_coord = movingAverage_filter(user_coord, 3) # any another low-pass filter can be applied here
    user_coord[:, 2] = real_world[:, 2]

    real_world_file =  "real_world_coords_from_"+acc_interp_file
    np.savetxt(real_world_file, real_world, delimiter=',', fmt="%.10f")
    user_file =  "real_world_coords_from_"+acc_interp_file
    np.savetxt(user_file, user_coord, delimiter=',', fmt="%.10f")
    