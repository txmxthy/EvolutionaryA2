

# Estimation of Distribution Algorithm [20 marks]

The 0-1 knapsack problem has been studied in project 1. 

* Given a set of M items and a bag, each item i has a weight wi and a value vi, and the bag has a capacity Q. 
* The knapsack problem is to select a subset of items to be placed into the bag so that the total weight of the selected items does not exceed the capacity of the bag, and the total value of the selected items is maximised. 
* The formal problem description can be written as follows.


* _max v1x1 + v2x2 + · · · + vM xM , (1)_
* _s.t. w1x1 + w2x2 + · · · + wM xM ≤ Q, (2)_
* _xi ∈ {0, 1}, i = 1, ..., M, (3)_

where xi = 1 means that item i is selected, and xi = 0 means that item i is not selected.

In this question, you are given the following three knapsack problem instances:

* 10 269: with 10 items and bag capacity of 269. The optimal value is 295.
* 23 10000: with 23 items and bag capacity of 10000. The optimal value is 9767.
* 100 1000: with 100 items and bag capacity of 995. The optimal value is 1514.

The format of each file is as follows:

* M Q
* v1 w1
* v2 w2
* ... ...
* vM wM

In other words, in the first line, the first number is the number of items, and the second number is the bag capacity. From the second line, each line has two numbers, where the former is the value, and the latter is the weight of the item.

2
Develop a simple Estimation of Distribution Algorithm (EDA), for example based on either the univariate marginal distribution algorithm (UMDA) or the population based incremental learning (PBIL) algorithm, to solve the above knapsack problem instances. You should

1. Determine the proper individual representation and explain the reasons.
2. Design the proper fitness function and justify its use.
3. Design and implement your EDA algorithm.
4. Set proper algorithm parameters, such as population size, individual selection criteria, and learning rate (subject to your choice of EDA variations).
5. For each knapsack problem instance, run your EDA implementation for 5 times with different random seeds. Present the mean and standard deviation of your EDA in the 5 runs.
6. Draw the convergence curve of your EDA implementation for each knapsack problem instance. The x-axis of your convergence curve represents the number of generations. The y-axis stands for the average fitness of the best solutions in the population of the x-th generation from the 5 runs. Discuss your convergence curve and draw your conclusions.

In the report, you should describe the details of your EDA algorithm design and implementation (including the overall algorithm outline, solution representation, statistics calculation, new solution generation, and algorithm parameters). You should also show the results (mean and standard deviation, and convergence curves), and make discussions and conclusions in your report.