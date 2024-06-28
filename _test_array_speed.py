from array import array
import numpy as np
from numba import njit

n = 1000

npa = np.random.random((n, n))
pyl = list(npa.flatten())
pya = array('f', pyl)


def access_npa(npa, idx1, idx2):
    return npa[np.array([idx1, idx2]).T]

@njit
def access_npa_njit(npa, idx1, idx2):
    return [npa[i1, i2] for i1, i2 in zip(idx1, idx2)]

@njit
def access_pyl_njit(pyl, idx1, idx2, hlen):
    return [pyl[i1* hlen + i2] for i1, i2 in zip(idx1, idx2)]

# @njit
def access_pya_njit(pya, idx1, idx2, hlen):
    return [pya[i1* hlen + i2] for i1, i2 in zip(idx1, idx2)]

if __name__ == "__main__":
    import timeit
    size = 10_000
    number = 100
    idx1, idx2 = list(np.random.randint(0,n,size)), list(np.random.randint(0,n,size))
    print(timeit.timeit('access_npa(npa, idx1, idx2)', globals=globals(), number=number))
    print(timeit.timeit('access_npa_njit(npa, idx1, idx2)', globals=globals(), number=number, setup='access_npa_njit(npa, idx1, idx2)'))
    # print(timeit.timeit('access_pyl_njit(pyl, idx1, idx2, hlen=n)', globals=globals(), number=number, setup='access_pyl_njit(pyl, idx1, idx2, hlen=n)'))
    print(timeit.timeit('access_pya_njit(pya, idx1, idx2, hlen=n)', globals=globals(), number=number, setup='access_pya_njit(pya, idx1, idx2, hlen=n)'))

