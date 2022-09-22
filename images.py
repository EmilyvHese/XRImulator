"""
This file contains functions and classes related to the generation images in the form of a number of X-ray photons, 
each with location and time of arrival and energy. 
"""

import numpy as np

class image():
    """ 
    Class that defines a data format that all images to be processed by this package should follow. 
    It consists of a number of arrrays of specified size which should contain the energy, arrival time, and ... #TODO define further
    Note that this class only generates an empty image class of specified size.
    Generating the actual photons to fill it up should happen in a seperate function that manipulates an image class object. 
    """

    def __init__(self, size):
        """ Initiation function for the class. Generates arrays of the specified size for each parameter specified in the class docstring. """
        # Abbreviation of 'Times Of Arrival'.
        self.toa = np.zeros(size)
        self.energies = np.zeros(size)

        # Array containing coordinates of origin for each photon.
        self.loc = np.zeros((size, 2))

        #TODO add further relevant arrays. At least one that defines arrival location

        # Useful to have as shorthand
        self.size = size

#TODO add image generators (after full image data types and such are figured out.) 

def point_source(size, theta, phi, energy):
    im = image(size)
    im.energies[:] = energy
    im.loc[:] = np.array([theta, phi])
    im.toa = np.array([i for i in range(size)])

    return im

if __name__ == "__main__":
    im = point_source(10, 1, 2, 5)
    print(im.energies, '\n', im.loc, '\n', im.toa)