# Solution Overview
This solution to the Facility Location Optimization problem employs a novel two-stage genetic algorithm. Initially, it explores binary decisions regarding facility operations (open/close) using genetic algorithms, where the fitness function is derived from a preliminary exploration of continuous variables related to service levels. Subsequently, the algorithm focuses on these continuous variables, optimizing the distribution of services from facilities to demand points, considering constraints like demand fulfillment and facility capacity. This hierarchical approach effectively balances solution quality and computational efficiency in addressing the complex FLO problem.

# Method
As described, the problem is a mixed-integer linear programming (MILP) problem. It is "mixed" because it includes both continuous variables x_ij > 0 (representing the amount serviced from facility j to demand point i) and integer variables y_i ∈ {0,1}  (binary variables indicating whether a facility is established at location j or not). The objective function and constraints are linear with respect to these variables.

In this scenario, there are two related spaces our algorithm must explore, the continuous amount service per facility to the customer and the integer binary variable representing whether a facility is closed or not. Let's call the last space y_ i ∈ {0,1} the First Space and x_ij > 0 the Second Space. The proposed algorithm shall explore both these spaces in a hierarchical manner.

# Diagram of the solution 
![Screen Shot 2024-02-16 at 21 52 48](https://github.com/lucca11235/Facility-Location-Optimization/assets/91396656/54989ffe-e214-46b8-a433-74c7c9d52afc)
