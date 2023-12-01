import numpy as np


def constrained_sum_sample_pos(n: int = 3, total: int = 100):
    """
    Return a randomly chosen array of n positive integers summing to total. Each such array is equally likely to occur.
    URL: https://stackoverflow.com/questions/3589214/generate-random-numbers-summing-to-a-predefined-value

    :param n: number of splits.
    :param total: number to be split into n random values that sum up to total.
    :return: array of length n containing the split values that sum up to total.
    """

    dividers = sorted(np.random.choice(range(1, total), n - 1))
    return np.array([a - b for a, b in zip(dividers + [total], [0] + dividers)])


def generate_spectral_mixtures(sli, class_labels, nrows=1, ncols=10, nclasses=-1, total=100, random_total=False, replace=False):
    """
    Compute an array of shape (nrows * ncols * nbands) that contains random linear mixtures of randomly selected spectra from n classes from a spectral library array.
    n will be the number of classes in class_labels if not specified. If specified 0 < n <= number of classes in class_labels.

    :param sli: spectral library array (spectra * bands), scaled to reflectance [0..1].
    :param class_labels: np.array containing one label for each spectrum in sli, respectively.
    :param nrows: number of rows in array, default: 1.
    :param ncols: number of cols in array, default: 10.
    :param nclasses: number of classes, default: -1, indicating that n is number of classes.
    :param total: number to be split into n random values that sum up to total, default: 100 [%]. 
                  because mixed_spectra are later on divided by 100, changing total is similar to providing a shade fraction.
                  alternatively, shade can be provided as an extra endmember and class in the spectral library (constant zero).
    :param random_total: boolean to control if total should randomly be chosen between the defined total and 100 [%]. This is equally to randomly choosing a shade fraction.
    :param replace: boolean, input to np.random.choice. Whether the sample is with or without replacement. 
                    Default is False, meaning that a class can only occur once in each mixture.
                    If True, nclasses can be greater than number of classes in class_labels.
    :return: four arrays:
                 [0]: spectral mixture array (nrows * ncols * nbands).
                 [1]: fractions array (nrows * ncols * nclasses) in percentage [0..100].
                 [2]: array of indices of spectra (nrows * ncols * nclasses) used for mixing in spectral library.
                 [3]: array of indices of classes (nrows * ncols * nclasses) of spectra used for mixing in spectral library.
    """
    # if random_total is True and total is not 100 [%], a random total will be picked between the defined total and 100.
    if total < 100 and random_total==True:
        total = np.random.choice(np.arange(total,101))
    elif total > 100 and random_total==True:
        total = np.random.choice(np.arange(100,total+1))
    
    # get unique class labels and corresponding indices (translation of class labels to int)
    unique_classes, unique_class_indices = np.unique(class_labels, return_inverse=True)

    # set n number of classes if not provided or greater than number of classes
    if nclasses==-1 or nclasses > len(unique_classes):
        nclasses = len(unique_classes)
    
    # generate empty arrays to fill 
    class_indices = np.zeros((nrows, ncols, nclasses)).astype(int)
    endmember_indices = np.zeros((nrows, ncols, nclasses)).astype(int)
    fractions = np.zeros((nrows, ncols, nclasses)).astype(int)

    for row in range(class_indices.shape[0]):
        for col in range(class_indices.shape[1]):
            # randomly select class labels from the sli
            random_class_indices = np.random.choice(np.unique(unique_class_indices), size=nclasses, replace=replace)
            # save the selected class indexes
            class_indices[row, col] = random_class_indices
            # randomly generate fractions for each endmember
            fractions[row, col] = constrained_sum_sample_pos(nclasses, total)
            # then randomly select endmembers from the corresponding classes
            for idx, class_index in enumerate(random_class_indices):
                # and save the selected endmember indexes
                endmember_indices[row, col, idx] = np.random.choice(np.where(unique_class_indices==class_index)[0])

    # select the selected classes from the class list
    selected_classes = unique_classes[class_indices]
    # select the selected spectra from the library based on their indexes
    selected_spectra = sli[endmember_indices]
    # generate spectral mixtures based on the selected spectra and corresponding fractions and convert to units of reflectance [0..1]
    mixed_spectra = np.einsum('ijk,ijkl->ijl', fractions, selected_spectra) / 100

    return mixed_spectra, fractions, selected_spectra, selected_classes