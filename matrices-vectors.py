import numpy as np

# vectors
print("Vectors: ")
v = np.array([1,5,2,9])
u = np.array([3,6,0,-5])
print("v = ", v, "\n", "u = ", u, "\n")

# vector addition
print("Vector addition: ")
print("v+u = ", v+u, "\n")

# vector scaling
print("Vector scaling: ")
print("3v = ", 3*v, "\n")

# Dot-Product
print("Vector dot (inner) product: ")
print("u dot v = ", np.dot(u,v), "\n")
# np.dot(u,v) = u.dot(v)

# Length / L2 Norm of a vector
print("L2 norm of vector")
#%.2f means round up to 2 decimal places
print("||v|| = %.2f" % np.linalg.norm(v), "\n")
# np.sqrt(np.dot(v,v)) = np.linalg.norm(v)

# matrices
print("Matrices and Vectors: ")
M = np.array([ [1,9,-12], [15, -2, 0] ])
A = np.array([ [1, 1], [2, 1] ])
B = np.array([ [0, 8], [7, 11] ])
G = np.array([[2,0],[0,1],[1,1]])
u = np.array([1,2])
v = np.array([1,5,2,9])
print("Matrix A = \n", A)
print("Matrix B = \n", B)
print("Matrix M = ", M, "\n")
print("Matrix G = ", G, "\n")
print("Vector u = ", u, "\n")
print("Vector v = ", v, "\n")

print("Matrix Shape: ")
#columns by rows in matrix
print("M shape = ", M.shape, "\n")

# matrix addition
print("Matrix addition: ")
print("A+B = \n", A+B, "\n") # '\n' is the newline character

# matrix scaling
print("Matrix Scaling: ")
print("5B = \n", 5*B)

print("\n Matrix Multiplication of A and M")
# matrix multiplicaiton, c1, c2, and c3, all make the same output
C1 = np.matmul(A, M)
C2 = A.dot(M)
C3 = A@M
print("\n C1 = \n", C1)

# matrix transpose (multiple ways of doing it)
print("\n Transpose (switch columns and rows) Matrix M")
print("M^T = \n", np.transpose(M))
# print("M^T = \n", M.transpose())


# matrix inverse
print("\n Inverse of matrix A = \n", np.linalg.inv(A))

# v
print("\n Vector v shape: ", v.shape, " = ", v)

# row vector v
v = v.reshape(1,-1) # shape -1 in np.reshape means value is infered
print("\n Vector v reshaped as row: ", v.shape, " = ",  v)

# column vector v
v = v.reshape(-1,1)
print("\n Vector v reshaped as col: ", v.shape, " = \n",  v)

# matrix vector multiplication
print("\n Reshape vector u to be column and multiply with matrix G")
u = u.reshape(-1,1)
print("u", u.shape, " = \n", u)
print("Gu = \n", G.dot(u))


# inner product as matrix multiplication
print("\n Inner product of transposed v and v: ")
vdotv = np.matmul(np.transpose(v), v)
print(vdotv)

print("\n\nMore matrices --------------\n")

X = np.array([[1, 2, -1], [1, 0, 1]])
Y = np.array([[3, 1], [0, -1], [-2, 3]])
Z = np.array([1, 4, 6]).reshape(3, -1)
A = np.array([[1, 2], [3, 5]])
b = np.array([5, 13]).reshape(2, -1)

print("\n X: \n", X, "\n")
print("Y: \n", Y, "\n")
print("Z: \n", Z, "\n")
print("A: \n", A, "\n")
print("b: \n", b, "\n")

print("X*Y: \n", X@Y, "\n")
print("Y*X: \n", Y@X, "\n")
print("Z^T*Y: \n", Z.T@Y, "\n")
print("A^-1*b: \n", np.linalg.inv(A)@b)
