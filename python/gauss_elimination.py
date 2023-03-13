import numpy as np

def gauss_elimination(A, b):

    n = len(b)
    Ab = np.concatenate((A, b.reshape(n, 1)), axis=1)
    for i in range(n-1):
        max_row = i
        for j in range(i+1, n):
            if abs(Ab[j, i]) > abs(Ab[max_row, i]):
                max_row = j
        Ab[[i, max_row]] = Ab[[max_row, i]]
        for j in range(i+1, n):
            Ab[j] -= Ab[j, i]/Ab[i, i] * Ab[i]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, n] - np.dot(Ab[i, i+1:n], x[i+1:n]))/Ab[i, i]

    return x


A = np.array([[2, 6, 1], [1, 2, -1], [5, 7, -4]],
                dtype=float)
b = np.array([7, -1, 9],
                dtype=float)

x_gauss = gauss_elimination(A, b)
print("Gauss elimination: x = {:.2f} {:.2f} {:.2f}".format(
    x_gauss[0], x_gauss[1], x_gauss[2])) 
