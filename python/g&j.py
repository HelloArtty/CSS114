import numpy as np

def gauss_elimination(A, b):
    
    n = len(b)
    Ab = np.concatenate((A, b.reshape(n, 1)), axis=1)
    for i in range(n-1):
        max_row = i
        for j in range(i+1, n):
            if abs(Ab[j,i]) > abs(Ab[max_row,i]):
                max_row = j
        Ab[[i,max_row]] = Ab[[max_row,i]]
        for j in range(i+1, n):
            Ab[j] -= Ab[j,i]/Ab[i,i] * Ab[i]    

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i,n] - np.dot(Ab[i,i+1:n], x[i+1:n]))/Ab[i,i]
        
    return x

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



A = np.zeros((3,3), dtype=float)
b = np.zeros((3,1), dtype=float)

print("Enter the matrix A:")
for i in range(3):
    for j in range(3):
        A[i][j] = float(input())

print("Enter the matrix b:")
for i in range(3):
    b[i][0] = float(input())

# A = np.array([[1, 2, 9],[3, 2, 9],[2, 1, 9]], dtype=float)
# b = np.array([3, 6, 9], dtype=float)

x_gauss = gauss_elimination(A, b)
print("Gauss elimination: x = {:.2f} {:.2f} {:.2f}".format(x_gauss[0], x_gauss[1], x_gauss[2]))

x_gauss_jordan = gauss_jordan(A, b)
print("Gauss-Jordan elimination: x = {:.2f} {:.2f} {:.2f}".format(x_gauss_jordan[0], x_gauss_jordan[1], x_gauss_jordan[2]))
