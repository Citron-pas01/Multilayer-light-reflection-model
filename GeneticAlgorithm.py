import copy
import random
import numpy as np
import csv
import matplotlib.pyplot as plt
from input import pmBounds,pNvalue
from chromosome import Chromosome

class GeneticAlgorithm:

    def __init__(self, bounds, precision, pm, pc, pop_size, max_gen):
        """
        :param bounds: variable ranges
        :param precision:
        :param pm: mutationo probability
        :param pc: cross probability
        :param pop_size: group size
        :param max_gen: number of generations
        """
        self.bounds = bounds
        self.precision = precision
        self.pm = pm
        self.pc = pc
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.pop = []
        self.bests = [0] * max_gen
        self.g_best = 0

    def ga(self):
        # main function
        self.init_pop()
        best = self.find_best()['best']
        self.g_best = copy.deepcopy(best)
        y = [0] * self.max_gen
        X = []
        for i in range(self.max_gen):
            self.cross()
            self.mutation()
            self.select()
            best = self.find_best()['best']
            self.bests[i] = best
            if self.g_best.y < best.y:
                self.g_best = copy.deepcopy(best)
            y[i] = self.g_best.y
            print('Iteration {0}: Best Reflectance = {1}'.format(i,self.g_best.y))
            X.append(Chromosome.decoding(self.g_best,self.g_best.code))
            #print('The variable values are:', X[-1])
        return X, y

    def init_pop(self):
        # initialize group
        for i in range(self.pop_size):
            chromosome = Chromosome(self.bounds, self.precision)
            self.pop.append(chromosome)

    def cross(self):
        pop2 = []
        uppperbound = []
        for i in range(int(self.pop_size / 2)):

            if self.pc > random.random():
            # randon select 8 chromosomes in pops
                # index of the two parents to mate.
                i = 0
                j = 0
                while i == j:
                    i = random.randint(0, self.pop_size-1)
                    j = random.randint(0, self.pop_size-1)

                pop_i = self.pop[i]
                pop_j = self.pop[j]

            # get new code, off spring
                for k in range(0,len(pNvalue)):
                    pop2.append(random.randint(0, pop_i.length[k]-1) )
                for k in range(0,len(pNvalue)):     
                    pop_i.code[k]= pop_i.code[k][0:pop2[k]] + pop_j.code[k][pop2[k]:pop_i.length[k]]
                    pop_j.code[k] = pop_j.code[k][0:pop2[k]] + pop_i.code[k][pop2[k]:pop_i.length[k]] 

    def mutation(self):
        index = []
        for i in range(self.pop_size):
            if self.pm > random.random():
                pop = self.pop[i]
                # select mutation index
                for k in range(0,len(pNvalue)):
                    # pop.length[0]) represent pop.code_th_0_length
                    index.append(random.randint(0, pop.length[k]-1) )
                    i = pop.code[k][index[k]]
                    i = self.__inverse(i)
                    pop.code[k] = pop.code[k][:index[k]] + i + pop.code[k][(index[k]+1):]

    def select(self):
    # calculate fitness function
        sum_f = 0
        for i in range(self.pop_size): # pop_size?????
            self.pop[i].func() # what does self.pop[i].func() mean?

    # guarantee fitness > 0
        min = self.pop[0].y
        for i in range(self.pop_size):
            if self.pop[i].y < min:
                min = self.pop[i].y
        if min < 0:
            for i in range(self.pop_size):
                self.pop[i].y = self.pop[i].y + (-1) * min

    # roulette
        for i in range(self.pop_size):
            sum_f += self.pop[i].y
        p = [0] * self.pop_size
        for i in range(self.pop_size):
            p[i] = self.pop[i].y / sum_f
        q = [0] * self.pop_size
        q[0] = 0
        for i in range(self.pop_size):
            s = 0
            for j in range(0, i+1):
                s += p[j]
            q[i] = s

    # start roulette
        v = []
        for i in range(self.pop_size):
            r = random.random()
            if r < q[0]:
                v.append(self.pop[0])
            for j in range(1, self.pop_size):
                if q[j - 1] < r <= q[j]:
                    v.append(self.pop[j])
        self.pop = v

    def find_best(self):

        #Find the best unit in the current group
        best = copy.deepcopy(self.pop[0])
        for i in range(self.pop_size):
            if best.y < self.pop[i].y:
                best = copy.deepcopy(self.pop[i])
        return {'best':best}

    def __inverse(self, i):

        # replace 1 by 0, or replace 0 by 1 in the mutation period
        # param i: mutation position

        r = '1'
        if i == '1':
            r = '0'
        return r

if __name__ == '__main__':
    f = open('OptGAoutput.txt','w')
    BestSolution = []
    OutValue = []
    bounds = pmBounds
    precision = 100000
    algorithm = GeneticAlgorithm(bounds, precision, 0.03, 0.9, 20, 20)
    X, y = algorithm.ga()
    #print(X, y)
    best_match_idx = np.where(y == np.max(y))[0][0]
    BestSolution = X[best_match_idx]

    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111)
    iter = range(1,len(y)+1)
    ax.plot(iter, y, color="red")
    ax.set_xlabel('Interations',fontsize=12)
    ax.set_ylabel('Cumulative Reflectance',fontsize=12)
    ax.set_title('Optimized Reflectance via Genetic Algorithm')
    plt.show()
# File output

    f.write("Input\n")
    f.write(str(X))
    f.write("\n Sum \n")
    f.write(str(y))
    f.write("\n BestSolution \n")
    f.write(str(BestSolution))
    f.close()
# bounds: variable ranges
# precision:
# mutation probability
# cross probability
# group size
# number of generations