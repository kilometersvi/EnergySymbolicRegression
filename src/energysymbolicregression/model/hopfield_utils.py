import numpy as np
"""
In connectivity matrix Q, max eigenvalue and min eigenvalue correlate with max and min energy of the model. 
These functions work with this principle.
"""


def find_extreme_eigenvectors(Q):
    # Get eigenvectors correlating with maximum and minimum eigenvalues of Q

    # Calculate eigenvalues and eigenvectors of Q
    eigenvalues, eigenvectors = np.linalg.eig(Q)
    
    # Find the index of the maximum and minimum eigenvalues
    max_eigenvalue_index = np.argmax(eigenvalues)
    min_eigenvalue_index = np.argmin(eigenvalues)
    
    # Extract the corresponding eigenvectors
    V_max = eigenvectors[:, max_eigenvalue_index]
    V_min = eigenvectors[:, min_eigenvalue_index]
    
    # Normalize the eigenvectors (optional)
    V_max = V_max / np.linalg.norm(V_max)
    V_min = V_min / np.linalg.norm(V_min)
    
    return V_max, V_min


def find_extreme_internal_energy(Q):
    # Calculate eigenvalues of Q
    eigenvalues = np.linalg.eigvals(Q)
    
    # Find the maximum and minimum eigenvalues
    max_eigenvalue = np.max(eigenvalues)
    min_eigenvalue = np.min(eigenvalues)
    
    # Calculate the maximum and minimum internal energy
    max_internal_energy = -0.5 * max_eigenvalue
    min_internal_energy = -0.5 * min_eigenvalue
    
    return min_internal_energy, max_internal_energy


def closest_binary_eigenvector(eigenvector, k):
    """
    Find the closest binary eigenvector to the given eigenvector.
    
    Parameters:
        eigenvector (numpy.ndarray): The continuous eigenvector.
        k (int): The number of components to set to 1.
        
    Returns:
        numpy.ndarray: The closest binary eigenvector.
    """
    # Sort the indices of the eigenvector components in descending order
    sorted_indices = np.argsort(-eigenvector)
    
    # Create a binary vector of the same length as the eigenvector
    binary_vector = np.zeros_like(eigenvector)
    
    # Set the top k components to 1
    binary_vector[sorted_indices[:k]] = 1
    
    return binary_vector


def get_diff_of_centers(u):
    mean_activations_u = np.mean(u[u > 0.5]) if len(u[u > 0.5]) > 0 else 0
    mean_nonactivations_u = np.mean(u[u < 0.5]) if len(u[u < 0.5]) > 0 else 0
    return abs(mean_activations_u - mean_nonactivations_u)


def calc_internal_energy(Q, V):
    e = (-0.5 * np.dot(V.T, np.dot(Q, V)))
    return e


if __name__ == "__main__":
        
    # Example usage
    Q = np.array([[2, -1], [-1, 2]])  # Replace with your Q matrix
    V_max, V_min = find_extreme_eigenvectors(Q)
    print("V that maximizes energy:", V_max)
    print("V that minimizes energy:", V_min)

    k = len(V_max)/2

    closest_possible_V_max = closest_binary_eigenvector(V_max, k)
    closest_possible_V_min = closest_binary_eigenvector(V_min, k)


    print("binary V that maximizes energy:", closest_possible_V_max)
    print("binary V that minimizes energy:", closest_possible_V_min)

    max_E = calc_internal_energy(Q, closest_possible_V_max)
    min_E = calc_internal_energy(Q, closest_possible_V_min)

    print(f"domain of internal energy function with this Q is: [{min_E}, {max_E}]")