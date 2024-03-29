from facility_location_GA import FacilityLocationGA

test_case = {'capacities':{'A': 100, 'B': 150, 'C': 120},
             'opening_costs': {'A': 10, 'B': 12, 'C': 8},
             'demands': {1: 60, 2: 50, 3: 70, 4: 40},
             'transportation_costs':{
                'A': {1: 2, 2: 3, 3: 1, 4: 2},
                'B': {1: 3, 2: 2, 3: 2, 4: 1},
                'C': {1: 1, 2: 1, 3: 2, 4: 3}
             }}

test_case_2 = {'capacities':{'A': 200, 'B': 180, 'C': 160, 'D': 150, 'E': 140},
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

test_case_3 = {'capacities':{
                  'A': 200, 'B': 300, 'C': 150, 'D': 180, 'E': 220,
                  'F': 160, 'G': 140, 'H': 210, 'I': 130, 'J': 170
              },
              'opening_costs':{
                  'A': 20, 'B': 24, 'C': 15, 'D': 18, 'E': 22,
                  'F': 16, 'G': 14, 'H': 21, 'I': 13, 'J': 17
              },

              'demands':{
                  1: 110, 2: 95, 3: 120, 4: 85, 5: 100,
                  6: 90, 7: 80, 8: 115, 9: 105, 10: 75
              },
              'transportation_costs':{
                  'A': {1: 4, 2: 6, 3: 3, 4: 5, 5: 4, 6: 6, 7: 7, 8: 3, 9: 4, 10: 5},
                  'B': {1: 5, 2: 4, 3: 2, 4: 6, 5: 3, 6: 5, 7: 4, 8: 2, 9: 5, 10: 4},
                  'C': {1: 3, 2: 3, 3: 4, 4: 5, 5: 2, 6: 4, 7: 6, 8: 1, 9: 3, 10: 2},
                  'D': {1: 2, 2: 5, 3: 6, 4: 3, 5: 4, 6: 2, 7: 5, 8: 4, 9: 2, 10: 6},
                  'E': {1: 6, 2: 2, 3: 5, 4: 4, 5: 6, 6: 3, 7: 3, 8: 6, 9: 1, 10: 3},
                  'F': {1: 5, 2: 7, 3: 3, 4: 2, 5: 5, 6: 7, 7: 2, 8: 5, 9: 6, 10: 1},
                  'G': {1: 3, 2: 1, 3: 7, 4: 6, 5: 3, 6: 5, 7: 4, 8: 3, 9: 7, 10: 2},
                  'H': {1: 4, 2: 3, 3: 1, 4: 7, 5: 2, 6: 6, 7: 1, 8: 4, 9: 5, 10: 7},
                  'I': {1: 7, 2: 5, 3: 2, 4: 1, 5: 7, 6: 1, 7: 7, 8: 7, 9: 3, 10: 4},
                  'J': {1: 6, 2: 4, 3: 6, 4: 2, 5: 1, 6: 3, 7: 6, 8: 2, 9: 6, 10: 5}
              }
              }

ga = FacilityLocationGA(test_case)
ga.initialize_population(10)

ga.run(iterations = 5)
ga.report()