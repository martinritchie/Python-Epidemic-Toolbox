"""The Gillespie algorithm."""
# M. Ritchie June 2017

import numpy as np
import pandas as pd
import networkx as nx
import random


class Gillespie(object):
    """The Gillepsie agloritm for simulating epidemics."""

    def __init__(self, size=50, tau=5.0, gamma=1.0, dt=0.1, tmax=None):

        # Network model.
        self._A = nx.complete_graph(size)
        # Epidemic parameters.
        self._tau = tau  # Per link rate of infection.
        self._gamma = gamma
        # Time and population counts.
        self._T = [0.0]
        self._S = [size - 1]
        self._I = [1]
        self._R = [0]
        # The state of and rate experianced by each node.
        self._state = np.zeros((size,))
        self._rate = np.zeros((size,))
        # I0, initial seed.
        I0 = random.randint(0, size)
        self._state[I0] += 1
        self._rate[I0] = gamma
        # I0 now transmits infection to neighbours.
        neighbors = self._A.neighbors(I0)
        self._rate[neighbors] += tau
        self._cumulateRate = np.cumsum(self._rate)
        self._tmax = tmax

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

    def calcTime(self):
        """Compute the time until the next event."""
        rate = np.sum(self._rate)
        self._T.append(self._T[-1] + np.log(random.random()) / (-rate))
        return rate

    def calcEvent(self, rate):
        """Compute the next event, and return the node index."""
        Event = random.random() * rate
        return np.argmax(self._cumulateRate > Event)

    def calcStateRate(self, event):
        """Adjust the state of event's neighbors."""
        neighbors = self._A.neighbors(event)
        Snodes = np.where(self._state == 0)
        Sneigh = np.intersect1d(Snodes, neighbors)
        if self._state[event] == 0:  # Infection.
            self._rate[Sneigh] += self._tau
            self._rate[event] = self._gamma
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
