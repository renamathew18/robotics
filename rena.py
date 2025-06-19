import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu

matrix_size = 2 

if matrix_size == 2:

    a11 = 2 
    a12 = 1 
    a22 = 3 
    A = np.array([[a11, a12], [a21, a22]])

   
    b1 = 4 
    b2 = 5 
    b = np.array([b1, b2])

elif matrix_size == 3:
    
    a11 = 1 
    a12 = 2 
    a13 = 3 
    a21 = 4
    a22 = 5 
    a23 = 6 
    a32 = 8 
    a33 = 9 
    A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

 
    b1 = 10
    b2 = 11
    b3 = 12
    b = np.array([b1, b2, b3])

else:
    raise ValueError("Invalid matrix size. Choose 2 or 3.")

print("Matrix A:\n", A)
print("Vector b:\n", b)


print("\n--- Solving the system Ax = b ---")

try:
    x_gaussian = np.linalg.solve(A, b)
    print("\nSolution using Gaussian elimination (np.linalg.solve):\n", x_gaussian)
except np.linalg.LinAlgError:
    print("\nMatrix is singular. Cannot solve using Gaussian elimination.")

    A_inv = np.linalg.inv(A)
    x_inverse = np.dot(A_inv, b)
    print("\nSolution using inverse matrix method (A_inv * b):\n", x_inverse)
except np.linalg.LinAlgError:
    print("\nMatrix is singular. Cannot compute inverse.")


print("\n--- LU Decomposition ---")
P, L, U = lu(A)
print("Permutation matrix P:\n", P)
print("Lower triangular matrix L:\n", L)
print("Upper triangular matrix U:\n", U)
print("Verify: P @ L @ U =\n", P @ L @ U)


print("\n--- Transpose of A ---")
A_T = A.T
print(A_T)


if matrix_size == 2:
    print("\n--- Visualization of 2D System ---")
    plt.figure(figsize=(6, 6))

   
    x_vals = np.linspace(-10, 10, 400)

    
    if A[0, 1] != 0:
        y1_vals = (b[0] - A[0, 0] * x_vals) / A[0, 1]
        plt.plot(x_vals, y1_vals, label=f'{A[0,0]}x + {A[0,1]}y = {b[0]}')
    elif A[0, 0] != 0:
        
        plt.axvline(x=b[0] / A[0, 0], color='red', linestyle='--', label=f'{A[0,0]}x = {b[0]}')
    else:
      
        if A[0, 1] == 0 and b[0] == 0:
             print("Equation 1 is 0 = 0 (identity)")
        elif A[0, 1] == 0 and b[0] != 0:
             print(f"Equation 1 is {b[0]} = 0 (contradiction)")


    
    if A[1, 1] != 0:
        y2_vals = (b[1] - A[1, 0] * x_vals) / A[1, 1]
        plt.plot(x_vals, y2_vals, label=f'{A[1,0]}x + {A[1,1]}y = {b[1]}')
    elif A[1, 0] != 0:
        
        plt.axvline(x=b[1] / A[1, 0], color='green', linestyle='--', label=f'{A[1,0]}x = {b[1]}')
    else:
        
        if A[1, 1] == 0 and b[1] == 0:
             print("Equation 2 is 0 = 0 (identity)")
        elif A[1, 1] == 0 and b[1] != 0:
             print(f"Equation 2 is {b[1]} = 0 (contradiction)")

    try:
        plt.plot(x_gaussian[0], x_gaussian[1], 'ro', label='Solution')
    except NameError:
         print("Cannot plot solution as the matrix is singular.")
    except IndexError:
         print("Cannot plot solution for this configuration.") 

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('System of Linear Equations')
    plt.grid(True)
    plt.legend()
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(x_vals.min(), x_vals.max()) 
    plt.gca().set_aspect('equal', adjustable='box') 
    plt.show()
elif matrix_size == 3:
    print("\nVisualization is only supported for 2x2 systems.")