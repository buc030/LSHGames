import math
import numpy as np
import matplotlib.pyplot as plt
import itertools

#dim reduction d -> k
class JLGausssianMatrix:
    def __init__(self, epsilon, d, n):
        self.epsilon = epsilon
        self.n = n
        self.d = d
        self.k = self.getMinK()

        #construct the matrix:
        sigma = (1.0/k)**0.5
        self.matrix = sigma*np.random.randn(k, d)

    def getMinK(self):
        return math.ceil((4*( (self.epsilon**2)/2.0 - (self.epsilon**3)/3.0)**-1)*math.log(self.n))

    def map(self, x):
        return np.dot(self.matrix, x)


class GausssianPartitioning:
    def __init__(self, eta, c, d):
        self.init(eta, c, d)

    def init(self, eta, c, d):
        self.epsilon = d**(-0.5)
        self.eta = eta
        self.c = c
        self.d = d
        self.P = []
        #self.jl = JLGausssianMatrix(0.9, d, 1000)
        #self.d = self.jl.getMinK()

        #copy points
        for i in range(2**self.d):
            w = np.random.randn(self.d)
            self.P.append(w)
        print 'Done building'
    #incrementatly change c
    def set_c(self, c):
        self.c = c

    def h(self, p):
        p = (p / np.linalg.norm(p))*self.eta*self.c;
        i = 0
        for w in self.P:
            if np.dot(p, w) >= self.eta*self.c*self.epsilon*(self.d**0.5):
                return i
            i = i + 1

        return None

    def generateRandomPointOnSphere(self):
        r = self.eta*self.c
        #p = np.random.randn(self.d)
        p = np.random.uniform(0.0, 1.0, self.d)
        p = (p / np.linalg.norm(p))*r
        return p

    def test(self, num_test_points):
        num_of_distance_buckets = 10
        r = self.eta*self.c
        distance_to_collisions = {}  
        distance_to_non_collisions = {}  
        #max distance is 2cr, distance buckets:
        #(0 - 2cr/num_of_distance_buckets) , (2cr/num_of_distance_buckets, 4cr/num_of_distance_buckets) 
        #(i2cr/num_of_distance_buckets, (i+1)2cr/num_of_distance_buckets) 
        interval_size = (2*r)/float(num_of_distance_buckets)

        #the point must be on a sphere:
        for (p1, p2) in itertools.combinations([self.generateRandomPointOnSphere() for i in range(num_test_points)], 2):
        #for i in range(num_test_points):
            #(p1, p2) = (self.generateRandomPointOnSphere() ,self.generateRandomPointOnSphere() )
            bucket_num = int(np.linalg.norm(p1 - p2)/interval_size)

            if bucket_num not in distance_to_collisions:
                distance_to_collisions[bucket_num] = 0
            if bucket_num not in distance_to_non_collisions:
                distance_to_non_collisions[bucket_num] = 0
            #print "self.h(p2) = " + str(self.h(p2))
            if self.h(p1) == self.h(p2):
                distance_to_collisions[bucket_num] = distance_to_collisions[bucket_num] + 1
            else:
                distance_to_non_collisions[bucket_num] = distance_to_non_collisions[bucket_num] + 1

        plt.plot(
            [(bucket_num*interval_size)/float(self.c) for bucket_num in distance_to_collisions.keys()], \
            [distance_to_collisions[bucket_num]/float(distance_to_collisions[bucket_num] + distance_to_non_collisions[bucket_num]) for bucket_num in distance_to_collisions.keys()], 'o')
        plt.plot(
            [(bucket_num*interval_size)/float(self.c) for bucket_num in distance_to_collisions.keys()], \
            [distance_to_collisions[bucket_num]/float(distance_to_collisions[bucket_num] + distance_to_non_collisions[bucket_num]) for bucket_num in distance_to_collisions.keys()], \
            '--', label='c = ' + str(self.c))
        print 'Done testing'


        


n = 250

par = GausssianPartitioning(1, 1, 2000)
par.test(n)

par.set_c(1.5)
par.test(n)

par.set_c(2)
par.test(n)

plt.ylabel('Collisions ratio')
plt.xlabel('Distance / c')
plt.legend()
plt.show()