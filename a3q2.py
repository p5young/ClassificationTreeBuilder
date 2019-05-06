# CS486 Fall 2018
# Assignment 3, Question 2
# Phil Young
# 20491854

import numpy as np

# Factor object contains:
# names: the variables remaining in the factor (list of strings)
# arr: a multidimensional array representing the factor (ndarray)
class factorObj(object):
    # contructor
    def __init__(self, varNames=[] , array=None):
        # if no names given
        if (len(varNames) == 0):
            # names = empty list
            self.names = []
            # if no array and no names
            if (type(array) == type(None)):
                # Create identity factor (doesn't change any factor when multiplied)
                self.arr = np.array(1)
            # if array but no name
            else:
                # copy array
                self.arr = array.copy()
        # if names given
        else:
            # copy names
            self.names = varNames.copy()
            # if no array given
            if (type(array) == type(None)):
                # create n dimensional array of zeroes
                self.arr = np.zeros(2, dtype=float)
                for i in range(len(varNames) - 1):
                    self.arr = np.stack((self.arr, self.arr))
            # if name and array given
            else:
                # copy array
                self.arr = array.copy()
       
    # print() prints the multidimentional array in a pretty table
    def print(self):
        # catch factors with no variables
        if (len(self.names) == 0):
            print("This factor has no variables. value: {}".format(self.arr))
            print()
            return
        
        # recursivePrinter is called at the bottom - does heavy lifting
        def recursivePrinter(stringSoFar, array):
            if (array.ndim == 1):
                print(stringSoFar + 't | ' + str(array[1]))
                print(stringSoFar + 'f | ' + str(array[0]))
            else:
                recursivePrinter(stringSoFar + 't, ', array[1])
                recursivePrinter(stringSoFar + 'f, ', array[0])
            
        # print table head
        print(', '.join(self.names) + ' |')
        print('---' * len(self.names) + '----')
        # print table data
        recursivePrinter('', self.arr)
        print()


# Restricts the given variable in the given factor to the given value
def restrict(factor, variable, value):
    # change value to an int
    if (value):
        value = 1
    else:
        value = 0
        
    # find index of variable in factor
    try:
        index = factor.names.index(variable)
    except:
        # NOTHING TO RESTRICT - return unchanged factor
        return factor
    
    # create new factor object to avoid altering old one
    retval = factorObj(factor.names, factor.arr)
    # restrict variable in array
    retval.arr = np.take(retval.arr, value, index)
    # remove variable name from name list
    retval.names.remove(variable)
    
    return retval


# performs pointwise multiplication of two given factors
def multiply(factor1, factor2):
    # trivial case - some factor is just a number, no variables
    if (len(factor1.names) == 0 or len(factor2.names) == 0):
        return factorObj(factor1.names + factor2.names, factor1.arr * factor2.arr)
    
    # executes if factor1 and factor2 have any variables in common
    for i in factor1.names:
        for j in factor2.names:
            if i == j:
                # restrict i to be false
                false1 = restrict(factor1, i, False)
                false2 = restrict(factor2, i, False)
                iFalse = multiply(false1, false2)
                # restrict i to be true
                true1 = restrict(factor1, i, True)
                true2 = restrict(factor2, i, True)
                iTrue = multiply(true1, true2)
                
                retArr = np.array([iFalse.arr, iTrue.arr])
                # Note: iTrue.names and iFalse.names are identical
                # because they are both products of factor1 and factor2
                return factorObj([i] + iFalse.names, retArr) # exit call---
            
    # factors have no common variables, reduce dimension of factor1 and recurse
        
    # base case
    if (len(factor1.names) == 1):
        retArr = np.stack((factor2.arr * factor1.arr[0], factor2.arr * factor1.arr[1]))
        
    # restrict a variable from factor1 and recurse on both its values
    else:
        # name of variable we're restricting
        restrictMe = factor1.names[0]
        
        # recurse on first half
        array1 = restrict(factor1, restrictMe, False)
        array1 = multiply(array1, factor2)    
        # recurse on second half
        array2 = restrict(factor1, restrictMe, True)
        array2 = multiply(array2, factor2)
        # combine two halves
        retArr = np.array([array1.arr, array2.arr])
        
    return factorObj(factor1.names + factor2.names, retArr)

# Sums out a factor in a given variable
def sumout(factor, variable):
    # find index of variable to be summed out
    try:
        index = factor.names.index(variable)
    except:
        return factor # nothing to sum out, return unchanged factor
    
    # copy list of variable names and remove the summed out one
    varNames = factor.names.copy()
    varNames.remove(variable)
    # return new object
    return factorObj(varNames, factor.arr.sum(axis=index))

# Normalizes a factor by dividing each entry by the sum of all
def normalize(factor):
    total = factor.arr.sum()
    return factorObj(factor.names.copy(), factor.arr / total)

# Computes Pr(queryVariables|evidenceList) by the variable elimination algorithm
def inference(factorList, queryVariables, hiddenVariables, evidenceList):
    
    # part (i)
    print("STEP 1: RESTRICT FACTORS:")
    # restrict the factors in factorList according to the evidence in evidenceList
    for index, factor in enumerate(factorList):
        for var, value in evidenceList.items():
                # does nothing if var isn't a variable of factorList[index]
                factorList[index] = restrict(factorList[index], var, value)
                
                # Print out this step
                if factor is not factorList[index]:
                    print("restrict {} to be {} in:".format(var, value))
                    factor.print()
                    factorList[index].print()
                    factor = factorList[index]
        
    # part (ii)
    print("STEP 2: SUM OUT HIDDEN VARIABLES:")
    # sum out each hidden variable from the product of the factors containing that variable
    for var in hiddenVariables: # iterate over all hidden variables in order
        print("Summing out {}:".format(var))
        indices = []
        # find factors which contain hidden variable
        for index, factor in enumerate(factorList):
            if var in factor.names:
                indices.append(index)
        
        print("multiply these factors together (they contain {}):".format(var))
        product = factorObj()   # product of all factors containing hidden variable
        # multiply factors which contain hidden variable
        for index in indices:
            product = multiply(product, factorList[index])
            factorList[index].print()
            
        print("to produce this product:")
        product.print()
        
        # remove factors which contain hidden variable from factorList
        for index in reversed(indices):
            del factorList[index]
        
        # sum out the product and add to factorList
        product = sumout(product, var)
        
        print("then sum out {} to produce:".format(var))
        product.print()
        factorList.append(product)
        
    # part (iii)
    print("STEP 3: MULTIPLY:")
    # multiply entire factorList into one result
    print("Multiply entire factor list into one result. Printing factor list:")
    result = factorObj()
    for factor in factorList:
        result = multiply(result, factor)
        factor.print()
        
    print("product is:")
    result.print()
    
    # part (iiii)
    print("STEP 4: NORMALIZE:")
    # normalize result
    result = normalize(result)
    
    print("normalize product to get final result:")
    result.print()

    return result

# Inputs the data from the modified holmes problem problem
def getFactorList():
    
    f1 = factorObj(['b'], np.array([0.9, 0.1]))
    
    f2 = factorObj(['e'], np.array([0.95, 0.05]))
    
    f3 = factorObj(['a','e','b'])
    f3.arr[1,1,1] = 0.95
    f3.arr[1,1,0] = 0.1
    f3.arr[1,0,1] = 0.9
    f3.arr[1,0,0] = 0.05
    f3.arr[0,1,1] = 0.05
    f3.arr[0,1,0] = 0.9
    f3.arr[0,0,1] = 0.1
    f3.arr[0,0,0] = 0.95
    
    f4 = factorObj(['w','a'])
    f4.arr[1,1] = 0.8
    f4.arr[1,0] = 0.4
    f4.arr[0,1] = 0.2
    f4.arr[0,0] = 0.6
    
    f5 = factorObj(['g','a'])
    f5.arr[1,1] = 0.4
    f5.arr[1,0] = 0.05
    f5.arr[0,1] = 0.6
    f5.arr[0,0] = 0.95
    
    result = [f1, f2, f3, f4, f5]
    
    print("factors used are:")
    for factor in result:
        factor.print()
    
    return result

def main():
    
    print("Question 2.1:")
    print("-------------------------------------------------------------------------")
    # build factor list from holmes problem data (hard coded)
    factorList = getFactorList()
    
    queryVariables = ['g']
    
    hiddenVariables = ['a','b','e']
    
    evidenceList = {'w': True}
    
    print("Determining P(G | W):")
    print()
    
    result1 = inference(factorList, queryVariables, hiddenVariables, evidenceList)
    print("Therefore P(G | W) = {}".format(result1.arr[1]))
    print("-------------------------------------------------------------------------")
    # build factor list from holmes problem data (hard coded)
    factorList = getFactorList()
    
    queryVariables = ['g']
    
    hiddenVariables = ['a','b','e']
    
    evidenceList = {'w': False}
    
    print("Determining P(G | ¬W):")
    print()
    
    result2 = inference(factorList, queryVariables, hiddenVariables, evidenceList)
    print("Therefore P(G | ¬W) = {}".format(result2.arr[1]))
    print("And we know from before P(G | W) = {}".format(result1.arr[1]))
    print("Therefore P(G | W) != P(G | ¬W), as required.")
    print("\n\n")
    
    
    print("Question 2.2:")
    print("-------------------------------------------------------------------------")
    # build factor list from holmes problem data (hard coded)
    factorList = getFactorList()
    
    queryVariables = ['b']
    
    hiddenVariables = ['e']
    
    evidenceList = {'w': True, 'g': True, 'a': True}
    
    print("Determining P(B | W ^ G ^ A):")
    print()
    
    result1 = inference(factorList, queryVariables, hiddenVariables, evidenceList)
    print("Therefore P(B | W ^ G ^ A) = {}".format(result1.arr[1]))
    print("-------------------------------------------------------------------------")
    # build factor list from holmes problem data (hard coded)
    factorList = getFactorList()
    
    queryVariables = ['b']
    
    hiddenVariables = ['w','g','e']
    
    evidenceList = {'a': True}
    
    print("Determining P(B | A):")
    print()
    
    result2 = inference(factorList, queryVariables, hiddenVariables, evidenceList)
    print("Therefore P(B | A) = {}".format(result2.arr[1]))
    print("And we know from before P(B | W ^ G ^ A) = {}".format(result1.arr[1]))
    print("Therefore P(B | W ^ G ^ A) == P(B | A), as required.")
    print("\n\n")
    
    
    print("Question 2.3:")
    print("-------------------------------------------------------------------------")
    print("P(B | W ^ G ^ A) was determined in the last problem.")
    print("Because AND operations are commutitive, P(B | A ^ G ^ W) == P(B | W ^ G ^ A).")
    print("So I am going to use the result from last problem and not calculate again it here.")
    print()
    print("Therefore P(B | A ^ G ^ W) = {}".format(result1.arr[1]))
    print("-------------------------------------------------------------------------")
    # build factor list from holmes problem data (hard coded)
    factorList = getFactorList()
    
    queryVariables = ['b']
    
    hiddenVariables = ['g','a','e']
    
    evidenceList = {'w': True}
    
    print("Determining P(B | W):")
    print()
    
    result2 = inference(factorList, queryVariables, hiddenVariables, evidenceList)
    print("Therefore P(B | W) = {}".format(result2.arr[1]))
    print("And we know from before P(B | A ^ G ^ W) = {}".format(result1.arr[1]))
    print("Therefore P(B | A ^ G ^ W) != P(B | W), as required.")
    print("\n\n")
    
    
    print("Question 2.4:")
    print("-------------------------------------------------------------------------")
    # build factor list from holmes problem data (hard coded)
    factorList = getFactorList()
    
    queryVariables = ['e']
    
    hiddenVariables = ['w','g']
    
    evidenceList = {'a': True, 'b': True}
    
    print("Determining P(E | A ^ B):")
    print()
    
    result1 = inference(factorList, queryVariables, hiddenVariables, evidenceList)
    print("Therefore P(E | A ^ B) = {}".format(result1.arr[1]))
    print("-------------------------------------------------------------------------")
    # build factor list from holmes problem data (hard coded)
    factorList = getFactorList()
    
    queryVariables = ['e']
    
    hiddenVariables = ['w','g','b']
    
    evidenceList = {'a': True}
    
    print("Determining P(E | A):")
    print()
    
    result2 = inference(factorList, queryVariables, hiddenVariables, evidenceList)
    print("Therefore P(E | A) = {}".format(result2.arr[1]))
    print("And we know from before P(E | A ^ B) = {}".format(result1.arr[1]))
    print("Therefore P(E | A ^ B) != P(E | A), as required.")
    print("\n\n")
    
    
if __name__ == '__main__': main()