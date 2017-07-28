"""
The Gillespie algorithm.
M. Ritchie June 2017. 
This has not yet been compared to ODEs.
"""

import networkx as nx, numpy as np, pandas as pd
import random


class Gillespie(object):
    """
    The Gillespsie algorithm for simulating SIR epidemics.
    """

    def __init__(self, size=10, tau=5.0, gamma=1.0, I0=1, dt=0.01,
                 tmax=None):

        # Number of nodes.
        self._N = size
        # Network model.
        self._A = nx.fast_gnp_random_graph(size, 10 / size)
        # Epidemic parameters.
        self._tau = tau  # Per link rate of infection.
        self._gamma = gamma
        self._I0 = I0
        # Time and population counts.
        self._T = [0.0]
        self._S = [size - 1]
        self._I = [1]
        self._R = [0]

        # The state of and rate experienced by each node.
        self._state = np.zeros((size,))
        self._rate = np.zeros((size,))

        # Set for lookups.
        self._Snodes = set(np.arange(size))

        # I0, initial seed.
        self.setInitialSeed()

        self._cumulateRate = np.cumsum(self._rate)

        self._tmax = tmax
        # Used to quantise data.
        self._dt = dt

    def __call__(self, tau=5.0):

        # Network model.
        self._A = nx.fast_gnp_random_graph(self._N, 5 / self._N)

        self._tau = tau

        # The state of and rate experienced by each node.
        self._state = np.zeros((self._N,))
        self._rate = np.zeros((self._N,))

        # Set for lookups.
        self._Snodes = set(np.arange(self._N))
        self.setInitialSeed()
        self._cumulateRate = np.cumsum(self._rate)

        return self.stepUntil()

    @property
    def popCounts(self):
        """Return the population counts."""
        return(self._S, self._I, self._R)

    @property
    def Time(self):
        """Return the raw time."""
        return(self._T)

    @property
    def rates(self):
        """Return the rates of infection and recovery."""
        return(self._tau, self._gamma)

    def setInitialSeed(self):
        # I0, initial seed.
        I0 = np.random.choice(self._N, self._I0)

        self._state[I0] += 1
        self._rate[I0] = self._gamma
        for node in I0:
            self._Snodes.remove(node)
            neighbors = self._A.neighbors(node)
            self._rate[neighbors] += self._tau

    def step(self):
        """Advance the simulation by a single time step."""
        rate = self.calcTime()
        event = self.calcEvent(rate)
        self.calcStateRate(event)

    def stepUntil(self, time=None):
        """Simulate until extinction or a preset time."""
        if time == None:
            while self._I[-1] > 0:  # Exctinction.
                self.step()
        else:
            while self.Time[-1] < time:  # Or a preset time.
                self.step()
        return self.digitise()

    def calcTime(self):
        """Compute the time until the next event."""
        rate = np.sum(self._rate)
        self._T.append(self._T[-1] + np.log(random.random()) / (-rate))
        return rate

    def calcEvent(self, rate):
        """Compute the next event, and return the node index."""
        Event = random.random() * rate
        return np.searchsorted(self._cumulateRate, Event)

    def calcStateRate(self, event):
        """Adjust the state of event's neighbors."""
        neighbors = self._A.neighbors(event)
        Sneigh = [n for n in neighbors if n in self._Snodes]
        if self._state[event] == 0:  # Infection.
            self._rate[Sneigh] += self._tau
            self._rate[event] = self._gamma
            self._Snodes.remove(event)
            self._S.append(self._S[-1] - 1)
            self._I.append(self._I[-1] + 1)
            self._R.append(self._R[-1])
        elif self._state[event] == 1:  # Recovery.
            self._rate[Sneigh] -= self._tau
            self._rate[event] = 0
            self._S.append(self._S[-1])
            self._I.append(self._I[-1] - 1)
            self._R.append(self._R[-1] + 1)
        self._state[event] += 1
        self._cumulateRate = np.cumsum(self._rate)

    def digitise(self):
        """Compute population counts over evenly spaced time steps."""
        t = 0
        tidx = 0
        idx = 0
        S, I, R, T, = list(), list(), list(), list()
        while t < self._T[-1]:
            while t > self._T[idx]:
                idx += 1

            S.append(self._S[idx])
            I.append(self._I[idx])
            R.append(self._R[idx])
            T.append(t)
            t += self._dt
            tidx += 1

        return pd.DataFrame({"S": S, "I": I, "R": R, "T": T})