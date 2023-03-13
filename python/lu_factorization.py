import numpy as np

def lu_factorization(A, b):
    n,m = A.shape
    L = np.zeros((n, n))
    U = np.zeros((n, m))
    
    for k in range(n):
        L[k, k] = 1.0
        for j in range(k, m):
            U[k, j] = A[k, j] - np.dot(L[k, :k], U[:k, j])
        for i in range(k+1, n):
            L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]
    
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i,:i], y[:i])) / L[i,i]
        
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i,i+1:], x[i+1:])) / U[i,i]
    
    return L, U, x

A = np.array([[1, 1, -1], 
            [1, -2, 3], 
            [2, 3, 1]])
b = np.array([4, -6, 7])

L ,U ,x = lu_factorization(A, b)

print("Lower triangular matrix L:")
print(L)

print("Upper triangular matrix U:")
print(U)

print("Solution vector x:")
print(x)

