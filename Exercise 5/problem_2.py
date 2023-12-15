import numpy as np
import matplotlib.pyplot as plt

MAX_ELEMENT_SIZE_IN_INPUT = 10

def matrix_mult_verify(A, B, C):
    """
    Verifies matrix multiplication result using Freivalds’ algorithm.

    Parameters:
    - A (numpy.ndarray): The first input matrix.
    - B (numpy.ndarray): The second input matrix.
    - C (numpy.ndarray): The matrix to be verified against the multiplication of A and B.

    Returns:
    - bool: True if the multiplication is verified, False otherwise.

    Freivalds’ algorithm checks the correctness of matrix multiplication without directly
    computing the actual product. It takes three matrices 'A', 'B', and 'C', and a random
    vector 'r' of 0s and 1s. It calculates 'Y = A * (B * r) - (C * r)' and checks if 'Y' 
    is a zero vector. If 'Y' is a zero vector, the multiplication is considered verified,
    and the function returns True; otherwise, it returns False.
    """
    n = len(A)
    r = np.random.randint(2, size=n)  # Generate a random vector of 0s and 1s
    
    # Calculate Y = A * (B * r) - (C * r)
    Y = A.dot(B.dot(r)) - C.dot(r)
    
    return np.array_equal(Y, np.zeros(n))  # Check if Y is a zero vector

def corrupt_matrix(C, num_corruptions):
    """
    Corrupts a given matrix by randomly altering specified elements.

    Parameters:
    - C (numpy.ndarray): The input matrix to be corrupted.
    - num_corruptions (int): The number of elements to be randomly altered.

    Returns:
    - numpy.ndarray: The corrupted matrix.

    This function randomly selects elements in the input matrix 'C' and alters them.
    It generates 'num_corruptions' random indices within the matrix and changes the
    values at these indices to random integers (selected from a range defined by 
    'MAX_ELEMENT_SIZE_IN_INPUT') that are different from the original value. It 
    ensures that the corrupted value is different from the original one.
    """
    n = len(C)
    
    # Randomly select indices to corrupt in matrix C
    indices = np.random.choice(n*n, num_corruptions, replace=False)
    C_flatten = C.flatten()
    
    # Corrupt selected indices
    for index in indices:
        # select random 2 choice to corrupt an element of C
        # it uses 2 choice in case, first matches the existing element 
        # it usees other choice
        random_choice = np.random.choice(MAX_ELEMENT_SIZE_IN_INPUT, 2, replace=False)
        if random_choice[0] != C_flatten[index]:
            C_flatten[index] = random_choice[0]
        else:
            C_flatten[index] = random_choice[1]
        
    return C_flatten.reshape(n, n)


def main():
    """
    Generates random matrices, corrupts them at varying degrees, and plots error rates.

    This function generates two random matrices 'A' and 'B' of size (n x n) with elements 
    ranging between 0 and 'MAX_ELEMENT_SIZE_IN_INPUT'. It computes the correct product 'C' 
    by multiplying 'A' and 'B' matrices using the dot product operation.

    It then proceeds to test different corruption levels on 'C' using the 'corrupt_matrix' 
    function and verifies the multiplication using 'matrix_mult_verify' based on Freivalds’ 
    algorithm. The error rates for different corruption levels are calculated and plotted 
    against the number of corruptions.
    """

    n = 10  # dimension of matrix
    # creates 2 random matrix with elements in between 0 and 10
    A = np.random.randint(MAX_ELEMENT_SIZE_IN_INPUT, size=(n, n))
    B = np.random.randint(MAX_ELEMENT_SIZE_IN_INPUT, size=(n, n))
    C = A.dot(B)  # Correct product

    # Maximum number of corruptions to test
    max_corruptions = 40  
    # Number of runs for each corruption level
    runs = 1000  

    error_rates = []
    corruption_levels = []
    expected_error_rate = []

    for num_corruptions in range(1, max_corruptions + 1):
        error_count = 0
        
        for i in range(runs):
            # Corrupt some entries in C
            corrupted_C = corrupt_matrix(C, num_corruptions)
            
            # Verify multiplication using Freivalds’ algorithm
            is_correct = matrix_mult_verify(A, B, corrupted_C)
            if not is_correct:
                error_count += 1

        error_rate = error_count / runs
        error_rates.append(error_rate)
        corruption_levels.append(num_corruptions)
        expected_error_rate.append(1 - 0.5**num_corruptions)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(corruption_levels, error_rates, marker='o', label='probability of finding error')
    plt.plot(corruption_levels, expected_error_rate, marker='', linestyle='--', color='red', label='Expected Error Rate (1 - 0.5^n)')
    plt.title('Probability of finding error vs. Number of Corruptions')
    plt.xlabel('Number of Corruptions')
    plt.ylabel('Probability of finding error')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('problem_2.pdf')

if __name__ == "__main__":
    main()
