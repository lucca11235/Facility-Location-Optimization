from first_space_GA import FirstSpaceGA
import numpy as np


class FacilityLocationGA(FirstSpaceGA):
    def __init__(self, facility_information):
        super().__init__(facility_information)
        self.facility_information = facility_information

    def routingGA(self,
                  size = 500,
                  iterations = 100,
                  mutation_rate = 0.15):

        open_facilities = np.where(self.best_individual == 1)[0]
        self.secondSpaceGA.optimize(open_facilities,
                                size = size,
                                iterations = iterations,
                                mutation_rate = mutation_rate,
                                verbose = True)


       
        print('\nSimulation Done')

    def run(self,iterations = 10):
        print("----EXPLORING FIRST SPACE----")
        # Explore first space
        super().run(iterations)

        # Exploit second space
        print("\n----EXPLOITING SECOND SPACE----")
        self.routingGA()

    def report(self):
        self.secondSpaceGA.report()

