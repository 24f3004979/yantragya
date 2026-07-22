'''
GAUSIAN ELIMINATION
Implementing simple algorithm to solve matrix equations with elementry row operations.

Approach
1. Find the pivot row
2. Forward Elimination
    > Target to get upper triangular matrix
    + Keep eliminating with elementary operations
3. Backward substitution
    > From lower up keep substituing values to find the solution

## Finding Lead row for matrix

1. Scan all columns for one
    - swap with first row : DONE
2. Scan for non-zero rows
    - multiply with reciprocal number to row
        terminate if float found in between
    - iterate for all to find lead

# BUG: Numerical Types and their opearations utilities are required for the operations
'''
import math

def find_lead(matrix):
    lead_column = [row[0] for row in matrix]

    # simple check
    for i in range(len(lead_column)):
        e = lead_column[i]
        if e == 1.0:
            print(f"Found a row with only ones easy termination")
            return matrix[i] # Lead Column

    # largest number swap
    for i in range(len(lead_column)):
        e = lead_column[i]
        scaler = 1/e 
        
        # Its not yet implemented at root fro these operations
        result = [float(elem) * scaler for elem in matrix[i]]
        print(f"After multiplying with scaller : {result}")
        no_float = True
        for _ in result:
            float_check = (float(math.floor(_)) == _) # absolute is equal to float
            print(f"For value {_} float check : {float_check} with abs {abs(_)}")

            if not(float_check):
                no_float = False
                print(f"{_} is float with scaller multiplication")
                break
        if float_check:
            print(f'Found the target lead row')
            return matrix[i]

    print(f"We cant find lead with swaping and scalling and pivoting for given matrix")


M = [[7,8],
     [2,2]]
r = find_lead(M)
print(r)
