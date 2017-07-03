#!/usr/bin/python3
# Realise a number of Gillespie simulations. 
# M. Ritchie, July 2017. 

from EpiBox import Gillespie

def main():
    model = Gillespie.Gillespie(size=10)
    model.stepUntil()
    model.quantise()
    print(model.popCounts)


if __name__ == '__main__':
    main()
