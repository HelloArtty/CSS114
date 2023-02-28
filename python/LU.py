import numpy as np

def lu_factorization(A):
    n,m = A.shape
    L = np.zeros((n, n))
    U = np.zeros((n, m))
    
    for k in range(n):
        L[k, k] = 1.0
        for j in range(k, m):
            U[k, j] = A[k, j] - np.dot(L[k, :k], U[:k, j])
        for i in range(k+1, n):
            L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]
    
    return L, U

# Example usage
A = np.array([[1, 3, 2], 
            [2, 1, 3], 
            [3, 2, 1]])

# A = np.array([[1, 2,3 ], 
#             [4, 5, 6], 
#             [7, 8, 9]])

# A = np.array([[1, 9,5 ], 
#             [5, 1, 8], 
#             [9, 9, 1]])

# A = np.array([[1 ,2 ,3], 
#               [1, -2 ,1], 
#               [5 ,-2 ,-8]])

# A = np.array([[1 ,2 ,1], [2, 0 ,1], [0 ,3 ,1]])


L, U = lu_factorization(A)
print("Coefficient matrix:\n", A)
print("Lower matrix:\n", L)
print("Upper matrix:\n", U)
