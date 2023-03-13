import numpy as np

def gauss_jordan(A, b):
    n = len(b)
    Ab = np.concatenate((A, b.reshape(n, 1)), axis=1)

    for i in range(n):
        max_row = i
        for j in range(i+1, n):
            if abs(Ab[j, i]) > abs(Ab[max_row, i]):
                max_row = j
        Ab[[i, max_row]] = Ab[[max_row, i]]
        Ab[i] /= Ab[i, i]
        for j in range(n):
            if j != i:
                Ab[j] -= Ab[j, i] * Ab[i]
    x = Ab[:, n]
    return x


A = np.array([[1 ,4 ,9], [4, 9, 16], [9 ,16 ,25 ]], 
                dtype=float)
b = np.array([10, 14, 18], 
                dtype=float)

x_gauss_jordan = gauss_jordan(A, b)
print("Gauss-Jordan elimination: x = {:.2f} {:.2f} {:.2f}".format(x_gauss_jordan[0], x_gauss_jordan[1], x_gauss_jordan[2]))