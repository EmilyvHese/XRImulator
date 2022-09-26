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

def point_source(size, alpha, beta, energy):
    """
    Function that generates an image of a monochromatic point source according to some specifications.

    Parameters:

    size (int) = number of photons to generate from this source.\n
    alpha (float) = coordinate offset from zero pointing in x-direction (arcsec)\n
    beta (float) = coordinate offset from zero pointing in y-direction (arcsec)\n
    energy (float) = energy of photons to generate (KeV)\n
    """
    im = image(size)
    im.energies[:] = energy * 1.602177733e-16
    im.loc[:] = np.array([alpha, beta]) * 2 * np.pi / (3600 * 360)
    im.toa = np.array([i for i in range(size)])

    return im

def double_point_source(size, alpha, beta, energy):
    """
    Function that generates an image of two monochromatic point sources according to some specifications.

    Parameters:

    size (int) = number of photons to generate from the sources.\n
    alpha (list-like of floats) = coordinate offsets from zero pointing in x-direction (arcsec)\n
    beta (list-like of floats) = coordinate offsets from zero pointing in y-direction (arcsec)\n
    energy (list-like of floats) = energies of photons to generate (KeV)\n
    """
    im = image(size)
    for i in range(size):
        source = np.random.randint(0,2)
        im.energies[i] = energy[source] * 1.602177733e-16
        im.loc[i] = np.array([alpha[source], beta[source]]) * 2 * np.pi / (3600 * 360)
        im.toa[i] = i

    return im

if __name__ == "__main__":
    im = point_source(10, 1, 2, 5)
    print(im.energies, '\n', im.loc, '\n', im.toa)