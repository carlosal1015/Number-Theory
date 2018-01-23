import numba as nb
import numpy as np
import time

n = 335000000

@nb.njit("i8(i4[:])", locals=dict(
    n=nb.int32, s=nb.int64, i=nb.int32,
    j=nb.int32, q=nb.int32, f=nb.int32))

def summarum(phi):
    s = 0

    phi[1] = 1

    i = 2
    while i < n:
        if phi[i] == 0:
            phi[i] = i - 1

            j = 2

            while j * i < n:
                if phi[j] != 0:
                    q = j
                    f = i - 1

                    while q % i == 0:
                        f *= i
                        q //= i

                    phi[i * j] = f * phi[q]
                j += 1
        s += phi[i]
        i += 1
    return s

if __name__ == "__main__":
    s1 = time.time()
    a = summarum(np.zeros(n, np.int32))
    s2 = time.time()

    print(a)
    print("{}s".format(s2 - s1))
