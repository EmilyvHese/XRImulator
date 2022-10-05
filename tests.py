import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

import images
import instrument
import process
import analysis

def update_D_test(p = False):
    """ Test to verify whether the 'update_D' from instrument.py works correctly. """


    #TODO 
    """Fix for current setup"""


    test_i = instrument.interferometer(2, .6, 1, 1, 1, .5)

    initial_D = test_i.D
    test_array = np.array([True for i in range(10)])

    if p:
        print(initial_D)

    test_i.set_target_D(test_i.D * 2)
    for i in range(len(test_array)):
        test_i.update_D()
        test_array[i] = test_i.D == initial_D

        if p: 
            print(test_i.D)

    if test_array[-1]:
        print("update_D failed to update D over time.")
    else:
        print("update_D succesfully updated D.")
        print("Please check with print enabled whether values are correct.")
    
    return

def wobbler_test(wobble_I, p = False):
    """ Test for the wobbler() function in instrument.py. """

    #TODO 
    """Fix for current setup"""

    test_i = instrument.interferometer(2, .6, 1, 1, 1, .5)
    test_i.wobbler(wobble_I)

    if test_i.theta and test_i.phi:
        print('The wobbler changed theta succesfully.')
        print('Please check with p = True whether values are correct.')
    else:
        print('The wobbler failed.')

    if p:
        for i in range(10):
            print(test_i.theta, test_i.phi)
            test_i.wobbler(wobble_I)
        print(test_i.theta, test_i.phi)

def ps_test():
    image = images.point_source(int(1e6), 0.001, 0.00, 1.2)

    test_I = instrument.interferometer()
    test_I.add_baseline(1, 10, 300, 17000, 2, 1)

    start = time.time()
    test_data = process.process_image(test_I, image, 0)
    print('Processing this image took ', time.time() - start, ' seconds')

    analysis.hist_interferometer_data(test_data, 100)
    ft_x_data, ft_y_data = analysis.ft_data(test_data)
    analysis.plot_ft(ft_x_data, ft_y_data, 2)

def dps_test():
    image = images.double_point_source(10000, [-.001, .001], [0, 0], [1.2, 6])

    test_I = instrument.interferometer()
    test_I.add_baseline(1, 10, 300, 17000, 2, 1)
    # print(image.loc[2,0])

    start = time.time()
    test_data = process.process_image(test_I, image, 0)
    print('Processing this image took ', time.time() - start, ' seconds')

    analysis.hist_interferometer_data(test_data, 100)
    ft_x_data, ft_y_data = analysis.ft_data(test_data)
    analysis.plot_ft(ft_x_data, ft_y_data, 0)

def psmc_test():
    image = images.point_source_multichromatic(10000, 0, 0, [1.2, 6])

    test_I = instrument.interferometer()
    test_I.add_baseline(1, 10, 300, 17000, 2, 1)
    # print(image.loc[2,0])

    start = time.time()
    test_data = process.process_image(test_I, image, 0)
    print('Processing this image took ', time.time() - start, ' seconds')

    analysis.hist_interferometer_data(test_data, 100)
    ft_x_data, ft_y_data = analysis.ft_data(test_data)
    analysis.plot_ft(ft_x_data, ft_y_data, 0)


if __name__ == "__main__":
    ps_test()