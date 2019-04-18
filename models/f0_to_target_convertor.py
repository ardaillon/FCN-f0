#! /usr/bin/env python
"""
@author : Luc ardaillon
created : 17/01/2019
last revision : 10/04/2019

conversion from the ground truth f0 value to the target vector of pitch classes to be predicted by the models (for training)
"""

import numpy as np

def freq2cents(f0, f_ref = 10.):
    '''
    Convert a given frequency into its corresponding cents value, according to given reference frequency f_ref
    :param f0: f0 value (in Hz)
    :param f_ref: reference frequency for conversion to cents (in Hz)
    :return: value in cents
    '''
    c = 1200 * np.log2(f0/f_ref)
    return c

def cents2freq(cents, f_ref = 10.):
    '''
    conversion from cents value to f0 in Hz

    :param cents: pitch value in cents
    :param fref: reference frequency used for conversion
    :return: f0 value
    '''
    f0 = f_ref * 2 ** (cents / 1200)
    return f0

def f0_to_target_vector(f0, vecSize = 486, fmin = 30., fmax = 1000., returnFreqs = False):
    '''
    convert from target f0 value to target vector of vecSize pitch classes (corresponding to the values in cents_mapping) that is used as output by the CREPE model
    Unlike the original CREPE model, the first class corresponds to a frequency of 0 (for unvoiced segments).
    If the frequency is 0, all values are 0, except for the 1st value that is = 1.
    For all other cases, the values are gaussian blurred around the target_pitch class, with a maximum value of 1
    :param f0: target f0 value
    :return: target vector of vecSize pitch classes (regularly spaced in cents, from fmin to fmax)
    '''

    fmin_cents = freq2cents(fmin)
    fmax_cents = freq2cents(fmax)
    mapping_cents = np.linspace(fmin_cents, fmax_cents, vecSize)

    target_vec = np.zeros(vecSize)

    #get the idx corresponding to the closest pitch
    f0_cents = freq2cents(f0)

    #gaussian-blur the vector auround the taget pitch idx as stated in the paper :
    sigma = 25
    for i in np.arange(vecSize):
        target_vec[i] = np.exp(-((mapping_cents[i] - f0_cents)**2)/(2*(sigma**2)))

    if(returnFreqs):
        return target_vec, mapping_cents
    else:
        return target_vec

def f0_to_target_vector_crepe(f0):
    '''
    convert from target f0 value to target vector of 360 pitch classes (corresponding to the values in cents_mapping) that is used as output by the CREPE model
    Unlike the original CREPE model, the first class corresponds to a frequency of 0 (for unvoiced segments).
    If the frequency is 0, all values are 0, except for the 1st value that is = 1.
    For all other cases, the values are gaussian blurred around the target_pitch class, with a maximum value of 1
    :param f0: target f0 value
    :return: target vector of 360 pitch classes (1st class is for f0=0. All other correspond to frequencies spaced by 20 cents
    '''

    vecSize = 360
    mapping = (np.linspace(0, 7180, vecSize) + 1997.3794084376191)

    target_vec = np.zeros(vecSize)

    #get the idx corresponding to the closest pitch
    f0_cents = freq2cents(f0)
    sigma = 25

    #gaussian-blur the vector auround the taget pitch idx as stated in the paper :
    for i in np.arange(vecSize):
        target_vec[i] = np.exp(-((mapping[i] - f0_cents)**2)/(2*(sigma**2)))

    return target_vec

if __name__ == '__main__':

    f0 = 100.

    CREPE = False

    if(CREPE):
        tv = f0_to_target_vector_crepe(f0)
    else:
        tv = f0_to_target_vector(f0)

    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(tv)
    plt.show()

