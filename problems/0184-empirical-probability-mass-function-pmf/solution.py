def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    pmf = []
    n = len(samples)

    for element in set(samples):
        pmf.append((element, samples.count(element)/n))

    pmf.sort()
    
    return pmf
    pass