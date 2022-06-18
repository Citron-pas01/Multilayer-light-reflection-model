import math
import random
import numpy as np
from input import StackingReflection, pNvalue, mathfunction

class Chromosome:
    def __init__(self, bounds, precision):
        
        temp = np.zeros(len(pNvalue))
        length = np.zeros(len(pNvalue),dtype=int)
        code =  ['']*len(pNvalue)
        pN = []*len(pNvalue)
        self.y = 0
        self.pN = pNvalue
        self.code = code
        bounds = np.array(bounds)
        self.bounds = bounds
        self.length = length
        for m in range(0,len(pNvalue)):
            temp[m] = (bounds[m][1] - bounds[m][0]) * precision
            self.length[m] = math.ceil(math.log(temp[m], 2))

        self.rand_init()
        self.func()

    def rand_init(self):
        for k in range(0,len(pNvalue)):
            for i in range(self.length[k]):
                self.code[k] += str(random.randint(0, 1))

    def decode2(self, code):
       
        for m in range(0,len(pNvalue)):
            self.pN[m] = self.bounds[m][0] + int(code[m],2)*(self.bounds[m][1]-self.bounds[m][0])/(2**self.length[m] - 1)

    def decoding(self, code):
       # generate a value between each variable range
        for m in range(0,len(pNvalue)):
            self.pN[m] = self.bounds[m][0] + int(code[m],2)*(self.bounds[m][1]-self.bounds[m][0])/(2**self.length[m] - 1)
        return self.pN

    def func(self):
        self.y = StackingReflection(self)
        #self.y = mathfunction(self)