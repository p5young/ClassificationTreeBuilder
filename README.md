# ClassificationTreeBuilder
Builds a Classification Tree using the Variable Elimination Algorithm

Question 1a is in the file A3Q1a.pdf

Questions 1b, c, d, and e are solved by a3q1.py.
a3q1.py takes no command line arguments and runs as is.
a3q1.py sends its output to Question1Printout.txt.
To send its output to console, or another file:
    comment out or modify line 11 of a3q1.py.

All of the factors used for each question, the steps taken, the new factors
produced, and the final answers can be found in Question1Printout.txt.

If you need to know hiddenVariables, queryList, evidenceVariables,
or hiddenVariable order, they can be found in main().

Question 2 is solved by a3q2.py.
a3q2.py takes no command line arguments and runs as is.
a3q2.py is identical to a3q1.py except for:
    the hardcoded data found in getFactorList().
    the output file name found in line 11.
    the types of calls to inference() in main().
    
    ie: The algorithm and factorObject classes are unchanged.
    
All of the factors used for each question, the steps taken, the new factors
produced, and the final answers can be found in Question2Printout.txt.

If you need to know hiddenVariables, queryList, evidenceVariables,
or hiddenVariable order, they can be found in main().
