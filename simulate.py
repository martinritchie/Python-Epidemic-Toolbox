#!/usr/bin/python3
# Compute a number of Gillespie simulations.
# M. Ritchie, July 2017.

from EpiBox import Gillespie
import multiprocessing.pool, pandas as pd, seaborn, matplotlib.pyplot as plt
import numpy as np
from statistics import mean



class Simulate(object):
    '''Returns the averages of multiple Monte Carlo simulations.'''
    
    def __init__(self, size=10, tau = 2, gamma = 1, I0=1, dt=0.01, 
                    repetitions=10):

        # Number of nodes.
        self._size = size

        # Rates of infection and recovery respectively. 
        self._tau = tau
        self._gamma = gamma 
        self._I0 = I0
        
        # Used to digitise the Gillespie timesteps 
        self._dt = dt

        # Number of monte-carlo simulations to run. 
        self._repetitions = repetitions 
        self._results = pd.DataFrame()

        # Epidemic model (contains network model).
        self._model = Gillespie.Gillespie

    def multiSim(self):
        '''Compute multiple Monte Carlo simulation across all available 
        cores.''' 
        procpool = multiprocessing.pool.Pool()
        parameters = [] 
        for i in range(self._repetitions):
            parameters.append( (self._size, self._tau, self._gamma,
                                self._I0, self._dt) )

        results = list(procpool.starmap(self._model(), parameters))
        
        # Usefull for debugging.
        # self.plotCloud(results) 

        while results:
            # Do not include premature extinctions
            if max(results[-1]['I']) > 4*self._I0:
                self._results = self._results.add(results.pop(), 
                                    fill_value=0.0)
            else:
                self._repetitions -= 1
                results.pop()
        self._results = self._results.divide(self._repetitions)
        self.trimData()

    def trimData(self):
        '''Trim the ends of the Gillespie data.'''
        Rmax = np.argmax(self._results['R'])
        self._results = self._results[0:Rmax]

    def plotResults(self):
        '''Plot the population level averages of the simulations.'''
        plt.plot(self._results['T'], self._results['S'])
        plt.plot(self._results['T'], self._results['I'])
        plt.plot(self._results['T'], self._results['R'])
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.grid()
        plt.show()

    def plotCloud(self, results):
        '''Plot all individiual instances.'''
        # Usefull for Debugging
        for data in results:
            plt.plot(data['T'], data['S'],alpha=0.1, color='b')
            plt.plot(data['T'], data['I'],alpha=0.1, color='r')
            # plt.plot(data['T'], data['R'],alpha=0.41, color='g')
        plt.show()
    
    @property
    def getData(self):
        '''Return the simulated data'''
        return self._results


def main():
    simResult = Simulate(size=10000, I0=1, tau = 2, repetitions=10)
    simResult.multiSim()
    simResult.plotResults()

if __name__ == '__main__':
    main()