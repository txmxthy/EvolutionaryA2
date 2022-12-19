
# Evolutionary Programming and Differential Evolution Algorithms [20 marks]
* In this question, you have the task to implement 
* (1) the Evolutionary Programming (EP) algorithm and 
* (2) either the Differential Evolution (DE) or Evolution Strategy (ES) algorithms.
* You also need to apply the implemented algorithms to searching for the minimum of the following two functions, where D is the number of variables, i.e., x1, x2, ..., xD.

## Rosenbrock's function:
f1(x) = ∑D−1
i=1
(
100(x2
i − xi+1
)2 + (xi − 1)2)
, xi ∈ [−30, 30]

## Griewanks's function:
f2(x) = ∑D
i=1
x2
i
4000 − ∏D
i=1 cos ( xi√i
) + 1, xi ∈ [−30, 30]

For D = 20, do the following:
1. Implement any specific variation of the EP and DE/ES algorithms of your choice. Justify your choice of the EP and DE/ES algorithm variations implemented.
2. Choose appropriate algorithm parameters and population size, in line with your algorithm implementations.
3. Determine the fitness function, solution encoding, and stopping criterion in EP and DE/ES.
4. Since EP and DE/ES are stochastic algorithms, for each function, f1(x) or f2(x), repeat the experiments 30 times, and report the mean and standard deviation of your results, i.e., f1(x) or f2(x) values.
5. Analyze your results, and draw your conclusions.

Then for D = 50, solve the Rosenbrock’s function using the same algorithm settings (repeat 30 times). Report the mean and standard deviation of your results. Compare the performance of EP and DE/ES on the Rosenbrock’s function when D = 20 and D = 50. Analyze your results, and draw you conclusions.