"""
This file contains functions and classes related to the generation images in the form of a number of X-ray photons, 
each with location and time of arrival and energy. 
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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
        self.toa = np.zeros(size, dtype=int)
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
    for i in range(0, size):
        source = np.random.randint(0,2)
        im.energies[i] = energy[source] * 1.602177733e-16
        im.loc[i] = np.array([alpha[source], beta[source]]) * 2 * np.pi / (3600 * 360)
        im.toa[i] = i

    return im

def m_point_sources(size, m, alpha, beta, energy):
    """
    Function that generates an image of two monochromatic point sources according to some specifications.

    Parameters:

    size (int) = number of photons to generate from the sources.\n
    m (int) = number of point sources to generate.\n
    alpha (list-like of floats) = coordinate offsets from zero pointing in x-direction (arcsec)\n
    beta (list-like of floats) = coordinate offsets from zero pointing in y-direction (arcsec)\n
    energy (list-like of floats) = energies of photons to generate (KeV)\n
    """
    im = image(size)
    for i in range(0, size):
        source = np.random.randint(0,m)
        im.energies[i] = energy[source] * 1.602177733e-16
        im.loc[i] = np.array([alpha[source], beta[source]]) * 2 * np.pi / (3600 * 360)
        im.toa[i] = i

    return im

def point_source_multichromatic(size, alpha, beta, energy):
    """
    Function that generates an image of a multichromatic point source according to some specifications.

    Parameters:

    size (int) = number of photons to generate from this source.\n
    alpha (float) = coordinate offset from zero pointing in x-direction (arcsec)\n
    beta (float) = coordinate offset from zero pointing in y-direction (arcsec)\n
    energy (list-like of floats) = upper and lower bounds for energy of photons to generate (KeV)\n
    """
    im = image(size)
    for i in range(0, size):
        im.energies[i] = (np.random.random() * (energy[1] - energy[0]) + energy[0]) * 1.602177733e-16
        im.loc[i] = np.array([alpha, beta]) * 2 * np.pi / (3600 * 360)
        im.toa[i] = i

    return im

def generate_from_image(image_path, no_photons, img_scale):
    photon_img = image(no_photons)
    # Load the image and convert it to grayscale
    img = Image.open(image_path).convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    # img_array = np.zeros((10,10))
    # img_array[5,5] = 1
    # img_array[7,7] = 1

    pix_scale = np.array(img_array.shape)
    # print(pix_scale)

    # Generate a probability density function from the image
    pdf = img_array / np.sum(img_array)

    # histedimage, _, __ = np.histogram2d(pdf[:, 0], pdf[:, 1], pix_scale) 
    # plt.imshow(pdf)
    # plt.show()
    
    # Draw N samples from the probability density function
    photon_locations = np.random.choice(
        np.arange(img_array.size),
        size=no_photons,
        p=pdf.flatten()    
    )
    
    # Convert the flattened indices back into (x,y) coordinates
    photon_locations = np.column_stack(np.unravel_index(photon_locations, img_array.shape))
    # histedimage, _, __ = np.histogram2d(photon_locations[:, 0], photon_locations[:, 1], [np.arange(pix_scale[0] + .5), np.arange(pix_scale[1] + .5)]) 
    # plt.imshow(histedimage)
    # plt.show()

    photon_locations = (photon_locations - (pix_scale/2)) * img_scale / pix_scale * 2 * np.pi / (3600 * 360)

    # print(photon_locations, 0.0005 * 2 * np.pi / (3600 * 360))
    photon_img.loc = photon_locations

    for i in range(no_photons):
        photon_img.energies[i] = 1.2 * 1.602177733e-16
        photon_img.toa[i] = i 

    return photon_img, pix_scale

# if __name__ == "__main__":
    # im = point_source(10, 1, 2, 5)
    # print(im.energies, '\n', im.loc, '\n', im.toa)