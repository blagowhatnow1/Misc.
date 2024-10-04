import numpy as np
from scipy.linalg import eigh

# Step 1: Generate a random high-correlation matrix with noise
def generate_high_correlation_matrix(n, correlation_strength=0.95, random_noise_strength=0.05):
    # Start with a base correlation matrix filled with the correlation strength
    base_matrix = np.full((n, n), correlation_strength)
    
    # Ensure diagonal is 1 for a valid correlation matrix
    np.fill_diagonal(base_matrix, 1)
    
    # Add random noise for variability
    random_noise = np.random.uniform(-random_noise_strength, random_noise_strength, size=(n, n))
    
    # Symmetrize the matrix to maintain symmetry of the correlation matrix
    random_noise = (random_noise + random_noise.T) / 2
    
    # Add the noise to the base matrix
    noisy_matrix = base_matrix + random_noise
    
    # Ensure diagonal remains 1 after adding noise
    np.fill_diagonal(noisy_matrix, 1)
    
    return noisy_matrix

# Step 2: Apply Higham's algorithm to ensure the matrix is PSD
def higham_psd(matrix, tol=1e-8):
    # Eigen decomposition
    eigvals, eigvecs = eigh(matrix)
    
    # Replace negative eigenvalues with zeros (for PSD)
    eigvals[eigvals < tol] = 0
    
    # Reconstruct the matrix
    psd_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Rescale to ensure diagonals are exactly 1
    D = np.sqrt(np.diag(psd_matrix))
    psd_matrix = psd_matrix / np.outer(D, D)
    
    return psd_matrix

# Step 3: Generate a random high-correlation matrix and make it PSD
n = 10  # Number of variables
correlation_strength = 0.92  # High correlation base
random_noise_strength = 0.12  # Low noise to maintain high correlation

random_matrix = generate_high_correlation_matrix(n, correlation_strength, random_noise_strength)
psd_matrix = higham_psd(random_matrix)

# Step 4: Print the result
print("Generated Positive Semi-Definite High-Correlation Matrix:")
print(psd_matrix)

# Optionally, check if it's positive semi-definite
def is_positive_semidefinite(matrix):
    return np.all(np.linalg.eigvals(matrix) >= 0)

print("Is matrix positive semi-definite?", is_positive_semidefinite(psd_matrix))

# Calculate the average off-diagonal correlation
def average_off_diagonal(matrix):
    n = matrix.shape[0]
    return (np.sum(matrix) - np.trace(matrix)) / (n * (n - 1))

print("Average off-diagonal correlation:", average_off_diagonal(psd_matrix))

