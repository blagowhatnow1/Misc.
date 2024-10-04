import numpy as np
from scipy.linalg import eigh

# Step 1: Generate a random matrix and symmetrize it
def generate_random_correlation_matrix(n):
    # Generate a random matrix
    A = np.random.rand(n, n)
    
    # Symmetrize the matrix
    A = (A + A.T) / 2
    
    # Step 2: Force diagonal elements to 1 (for correlation matrix)
    np.fill_diagonal(A, 1)
    
    return A

# Step 3: Apply Higham's algorithm to ensure the matrix is PSD
def higham_psd(matrix, tol=1e-8):
    """
    Adjusts the matrix to be positive semi-definite using Higham's algorithm.
    Parameters:
    - matrix: Input matrix to be corrected.
    - tol: Tolerance for eigenvalue correction.
    """
    # Eigen decomposition (guaranteed to work for symmetric matrices)
    eigvals, eigvecs = eigh(matrix)
    
    # Replace negative eigenvalues or those below tolerance with zeros (for PSD)
    eigvals = np.clip(eigvals, tol, None)
    
    # Reconstruct the matrix from modified eigenvalues and eigenvectors
    psd_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Step 4: Rescale to ensure diagonals are exactly 1
    D = np.sqrt(np.diag(psd_matrix))
    psd_matrix = psd_matrix / np.outer(D, D)
    
    # Fix numerical issues with diagonal exactly being 1
    np.fill_diagonal(psd_matrix, 1)
    
    return psd_matrix

# Step 5: Generate a random matrix and make it a PSD correlation matrix
n = 10  # Number of variables
random_matrix = generate_random_correlation_matrix(n)
psd_matrix = higham_psd(random_matrix)

# Step 6: Print the result
print("Generated Positive Semi-Definite Correlation Matrix:")
print(psd_matrix)

# Optionally, check if it's positive semi-definite
def is_positive_semidefinite(matrix, tol=1e-8):
    # Eigenvalues must all be non-negative for the matrix to be PSD
    return np.all(np.linalg.eigvals(matrix) >= -tol)

print("Is matrix positive semi-definite?", is_positive_semidefinite(psd_matrix))

