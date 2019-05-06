# CS486 Fall 2018
# Assignment 3
# Phil Young
# 20491854

import numpy as np

import sys

# Comment out this line to print to stdout
sys.stdout = open('Question1Printout.txt', 'w')

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

# Inputs the data from the fido problem
def getFactorList():
    """
    fh - fido howls
    s - fido is sick
    b - food in fido's bowl
    m - full moon is out
    na - neighbour is away
    nh - neighbour's dog is howling
    
    NOTE TO MARKER:
        The assignment says to:
            "Indicate what queries (i.e., P r(variables|evidence)) you
            used to compute those probabilities."
        I will be using all these queries for Q1:b,c,d, and e
    P(s) = 0.05
    P(~s) = 0.95
    P(b | s) = 0.6
    P(~b | s) = 0.4
    P(b | ~s) = 0.1
    P(~b | ~s) = 0.9
    P(~na) = 0.7
    P(na) = 0.3
    P(m) = 1/28 = 0.035714285
    P(~m) = 27/28 = 0.964285714
    P(nh | ~m ^ ~na) = 0
    P(nh | ~m ^ na) = 0.5
    P(nh | m ^ ~na) = 0.4
    P(nh | m ^ na) = 0.8
    P(~nh | ~m ^ ~na) = 1
    P(~nh | ~m ^ na) = 0.5
    P(~nh | m ^ ~na) = 0.6
    P(~nh | m ^ na) = 0.2
    P(fh | s ^ m ^ nh) = 0.99
    P(fh | ~s ^ ~m ^ ~nh) = 0
    P(fh | s ^ ~m ^ ~nh) = 0.5
    P(fh | s ^ ~m ^ nh) = 0.75
    P(fh | s ^ m ^ ~nh) = 0.9
    P(fh | ~s ^ m ^ nh) = 0.65
    P(fh | ~s ^ m ^ ~nh) = 0.4
    P(fh | ~s ^ ~m ^ nh) = 0.2
    P(~fh | s ^ m ^ nh) = 0.01
    P(~fh | ~s ^ ~m ^ ~nh) = 1
    P(~fh | s ^ ~m ^ ~nh) = 0.5
    P(~fh | s ^ ~m ^ nh) = 0.25
    P(~fh | s ^ m ^ ~nh) = 0.1
    P(~fh | ~s ^ m ^ nh) = 0.35
    P(~fh | ~s ^ m ^ ~nh) = 0.6
    P(~fh | ~s ^ ~m ^ nh) = 0.8
    """
    print("Printing Factor List:")
    f1 = factorObj(['s'], np.array([0.95, 0.05]))
    print("P(S = s)")
    f1.print()
    
    f2 = factorObj(['na'], np.array([0.7, 0.3]))
    print("P(NA = na)")
    f2.print()
    
    f3 = factorObj(['m'], np.array([27/28, 1/28]))
    print("P(M = m)")
    f3.print()
    
    f4 = factorObj(['b','s'], np.array([[0.9, 0.4], [0.1, 0.6]]))
    print("P(B = b | S = s)")
    f4.print()
    
    f5 = factorObj(['nh','m','na'])
    print("P(NH = nh | M = m ^ NA = na)")
    f5.arr[1,1,1] = 0.8
    f5.arr[1,1,0] = 0.4
    f5.arr[1,0,1] = 0.5
    f5.arr[1,0,0] = 0
    f5.arr[0,1,1] = 0.2
    f5.arr[0,1,0] = 0.6
    f5.arr[0,0,1] = 0.5
    f5.arr[0,0,0] = 1
    f5.print()
    
    f6 = factorObj(['fh','s','m','nh'])
    print("P(FH = fh | S = s ^ M = m ^ NH = nh)")
    f6.arr[1,1,1,1] = 0.99
    f6.arr[1,1,1,0] = 0.9
    f6.arr[1,1,0,1] = 0.75
    f6.arr[1,1,0,0] = 0.5
    f6.arr[1,0,1,1] = 0.65
    f6.arr[1,0,1,0] = 0.4
    f6.arr[1,0,0,1] = 0.2
    f6.arr[1,0,0,0] = 0
    f6.arr[0,1,1,1] = 0.01
    f6.arr[0,1,1,0] = 0.1
    f6.arr[0,1,0,1] = 0.25
    f6.arr[0,1,0,0] = 0.5
    f6.arr[0,0,1,1] = 0.35
    f6.arr[0,0,1,0] = 0.6
    f6.arr[0,0,0,1] = 0.8
    f6.arr[0,0,0,0] = 1
    f6.print()
    
    return [f1, f2, f3, f4, f5, f6]

def main():
    
    print("Question 1b:")
    print("-------------------------------------------------------------------------")
    # build factor list from fido problem data (hard coded)
    factorList = getFactorList()
    
    queryVariables = ['fh']
    
    hiddenVariables = ['b', 'm', 'na', 'nh', 's']
    
    evidenceList = {}
    
    print("Determining Pr(FH = true) (no evidence given):")
    print()
    
    result = inference(factorList, queryVariables, hiddenVariables, evidenceList)
    print("Therefore the probability Fido howls, with no other knowledge, is:")
    print(result.arr[1])
    print("\n\n")
    
    print("Question 1c:")
    print("-------------------------------------------------------------------------")
    factorList = getFactorList()
    
    queryVariables = ['s']
    
    hiddenVariables = ['b', 'na', 'nh']
    
    evidenceList = {'fh': True, 'm': True}
    
    print("Determining Pr(S = true | FH = true ^ M = true):")
    print()
    
    result = inference(factorList, queryVariables, hiddenVariables, evidenceList)
    print("Therefore the probability Fido is sick (given howling and full moon) is:")
    print(result.arr[1])
    print("\n\n")
    
    print("Question 1d:")
    print("-------------------------------------------------------------------------")
    factorList = getFactorList()
    
    queryVariables = ['s']
    
    hiddenVariables = ['na', 'nh']
    
    evidenceList = {'b': True, 'fh': True, 'm': True}
    
    print("Determining Pr(S = true | B = true ^ FH = true ^ M = true):")
    print()
    
    result = inference(factorList, queryVariables, hiddenVariables, evidenceList)
    print("Therefore the probability Fido is sick (given bowl, howling, and fullmoon) is:")
    print(result.arr[1])
    print("\n\n")
    
    print("Question 1e:")
    print("-------------------------------------------------------------------------")
    factorList = getFactorList()
    
    queryVariables = ['s']
    
    hiddenVariables = ['nh']
    
    evidenceList = {'b': True, 'fh': True, 'm': True, 'na': True}
    
    print("Determining Pr(S = true | B = true ^ FH = true ^ M = true ^ NA = true):")
    print()
    
    result = inference(factorList, queryVariables, hiddenVariables, evidenceList)
    print("Therefore the probability Fido is sick (given bowl, howling, full moon, and neighbour away) is:")
    print(result.arr[1])
    
if __name__ == '__main__': main()