# Facility-Location-Optimization

Welcome to my repository dedicated to solving the Facility Location Optimization problem. This problem revolves around determining the most strategic locations for facilities (like warehouses, factories, or retail stores) to minimize costs and optimize service efficiency to a set of clients. It's an interesting space to test new techniques such as heuristics, linear programming, and even AI. This repository is my space to share some of the solutions I have developed or am currently developing.

## Problem Overview
The Facility Location Optimization problem involves several key components:

- *Facilities:* Places where goods are produced, stored, or sold.
- *Clients:* Endpoints that need the services or goods provided by the facilities.
- *Costs:* Includes various factors such as opening facilities, transportation, and operational expenses.
- *Objective:* To minimize the total cost, which may involve a combination of minimizing the distance clients are from facilities, minimizing the number of facilities, or minimizing the overall operational and transportation costs.

This problem can be modeled in various ways, including but not limited to the Simple Plant Location Problem (SPLP), the Capacitated Facility Location Problem (CFLP), and the Multi-Facility Location Problem (MFLP), each introducing different constraints and complexities.

As of now, the solution presented considers the CFLP model of the problem.

## Solutions
In this repository, I present a couple of solutions to the Facility Location Optimization problem, each designed with efficiency and scalability in mind. My approaches leverage a mix of traditional optimization techniques and modern computational methods, including:

- Heuristic algorithms for quick and effective solutions to complex problems.
- Exact algorithms for smaller instances where precision is paramount.
- Metaheuristic approaches that balance between solution quality and computational feasibility for larger instances.

## Mathematical formulation (CFLP)

![flo problem](https://github.com/lucca11235/Facility-Location-Optimization/assets/91396656/42a1e6bb-ae45-4351-bb1d-56b9d7051fd2)

## Results Example

Say the input of the problem is a dictionary like the following, where we have facilities named from 'A' to 'E' and demand points named from 1 to 6.

```
test =  {'capacities':{'A': 200, 'B': 180, 'C': 160, 'D': 150, 'E': 140},
        'opening_costs':{'A': 20, 'B': 25, 'C': 22, 'D': 18, 'E': 15},
        'demands':{1: 90, 2: 80, 3: 70, 4: 60, 5: 85, 6: 75}, 
        'transportation_costs':{
            'A': {1: 4, 2: 6, 3: 5, 4: 3, 5: 2, 6: 3},
            'B': {1: 5, 2: 3, 3: 4, 4: 2, 5: 6, 6: 5},
            'C': {1: 3, 2: 4, 3: 2, 4: 5, 5: 3, 6: 4},
            'D': {1: 6, 2: 5, 3: 3, 4: 4, 5: 5, 6: 2},
            'E': {1: 2, 2: 1, 3: 6, 4: 1, 5: 4, 6: 6}
        }
        }
```

The challenge escalates quickly due to the exponential growth of potential solutions. It involves not only deciding which facilities to open but also determining the distribution of units to meet the precise demands, constrained by each facility's capacity. Despite the complexity, the output from the developed genetic algorithm shows promising results, demonstrating efficient facility operation and transportation matrix. The algorithm found a pleasing combination, where not all facilities are opened and lots of the terms are 0, showing the algorithm understands some of facilities are not worth investing sending to some clients.

```
Best fitness: 1041.0

Facility A is open.
Facility C is open.
Facility E is open.

 Matrix of transportation:
      1     2     3     4     5     6
A   0.0   0.0   0.0  39.0  85.0  75.0
C  51.0   0.0  70.0   0.0   0.0   0.0
E  39.0  80.0   0.0  21.0   0.0   0.0
```

