import warnings
import numpy as np 

# ignore runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def affinity(pixel1, pixel2, sigma=4.):
    # define affinity function 
    return np.exp(np.linalg.norm(pixel1 - pixel2) / (2 * sigma**2))
