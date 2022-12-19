# INSTRUCTIONS:
In project 1 we have tried to use GP to evolve a single genetic program to solve the following
symbolic regression problem:
 f (x) = { 1/x + sin x , x > 0
           2/x + x^2 + 3.0 , x ≤ 0

In project 1, we assume that there is no prior knowledge about the target model.
In this project, the assumption is changed.

Instead of knowing nothing, we know that the target model is a piecewise function, with two sub-functions:
  f1(x) for x > 0 and f2(x) for x ≤ 0.
In other words, we know that the target function is:

f (x) = { f1(x), x > 0
          f2(x), x ≤ 0 

This question is to develop a Cooperative Co-evolution GP (CCGP) to solve this symbolic
regression problem.

The CCGP should contain two sub-populations, one for f1(x) and the other for f2(x).
You can use a GP library.

You should:
* Determine and describe the terminal set and the function set of each sub-population.
* Design the fitness function and the fitness evaluation method for each sub-population.
* Set the necessary parameters, such as:
  * sub-population size,
  * maximum tree depth,
  * termination criteria,
  * crossover and
  * mutation rates.
* Run the implemented CCGP for 5 times with different random seeds.

Report the best genetic programs (their structure and performance) of each of the 5 runs.
Present your observations and discussions and draw your conclusion