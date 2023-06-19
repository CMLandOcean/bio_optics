def spectral_angle(s1: np.array, s2: np.array):
    """
    Compute the spectral angle between two vectors in radians after Kruse et al. (1993) [doi.org/10.1016/0034-4257(93)90013-N].
    From https://pysptools.sourceforge.io/_modules/pysptools/distance/dist.html.

    :param s1: first spectrum
    :param s2: second spectrum

    :return:
    """
    try:
        angle =  math.acos(np.dot(s1,s2) / (np.sqrt(np.dot(s1,s1)) * np.sqrt(np.dot(s2,s2))))
    
    except ValueError:
        # python math don't like when acos is called with a value very near to 1
        return 0.0

    return angle   


def spectral_information_divergence(s1: np.array, s2: np.array):
    """
    Compute the spectral information divergence between two vectors in radians after Chang (2000) [doi.org/10.1109/18.857802].
    From https://pysptools.sourceforge.io/_modules/pysptools/distance/dist.html.

    :param s1: first spectrum
    :param s2: second spectrum

    :return:
    """
    p = (s1 / np.sum(s1)) + np.spacing(1)
    q = (s2 / np.sum(s2)) + np.spacing(1)
    return np.sum(p * np.log(p / q) + q * np.log(q / p)) 
    

def chebyshev_distance(s1: np.array, s2: np.array):
    """
    Compute the chebyshev distance between two vectors.
    From https://pysptools.sourceforge.io/_modules/pysptools/distance/dist.html.
    """
    return np.amax(np.abs(s1 - s2))