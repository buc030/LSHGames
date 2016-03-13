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
        sigma = (1.0/self.k)**0.5
        self.matrix = sigma*np.random.randn(self.k, d)

    def getMinK(self):
        return math.ceil((4 * math.log(self.n)) / (self.epsilon**2 / 2.0 - self.epsilon**3 / 3.0))

    def map(self, x):
        return np.dot(self.matrix, x)



class LSH:
    def __init__(self, d):
        self.d = d
        self.num_of_distance_buckets = 20

    def h(self, p):
        raise NotImplementedError('You need to define a h method!')

    def getPointsBound(self):
        raise NotImplementedError('You need to define a getPointsBound method!')

    def norm(self, p):
        raise NotImplementedError('You need to define a norm method!')



    #generate two random points in a random distance between d1 and d2
    def generatePointsInDistance(self, d1, d2):
        distance = np.random.uniform(0, d2 - d1) + d1
        dir = np.random.uniform(0, 1, self.d)
        dir = dir / self.norm(dir)

        p1 = np.random.uniform(0, self.getPointsBound(), self.d)
        p2 = p1 + dir*distance
        return [p1, p2]


    def test(self, num_test_points):

        r = self.getPointsBound()
        self.distance_to_collisions = {}  
        self.distance_to_non_collisions = {}  
        #max distance is 2cr, distance buckets:
        #(0 - 2r/num_of_distance_buckets) , (2r/num_of_distance_buckets, 4r/num_of_distance_buckets) 
        #(i2r/num_of_distance_buckets, (i+1)2r/num_of_distance_buckets) 
        interval_size = (2*r)/float(self.num_of_distance_buckets)
        
        #the point must be on a sphere:
        #for (p1, p2) in itertools.combinations([self.generateRandomPoint() for i in range(num_test_points)], 2):
        for i in range(self.num_of_distance_buckets):
            for j in range(num_test_points):
                [p1, p2] = self.generatePointsInDistance((i)*interval_size, (i + 1)*interval_size)
                bucket_num = int(self.norm(p1 - p2)/interval_size)

                if bucket_num not in self.distance_to_collisions:
                    self.distance_to_collisions[bucket_num] = 0
                if bucket_num not in self.distance_to_non_collisions:
                    self.distance_to_non_collisions[bucket_num] = 0
                #print "self.h(p2) = " + str(self.h(p2))
                if self.h(p1) == self.h(p2):
                    self.distance_to_collisions[bucket_num] = self.distance_to_collisions[bucket_num] + 1
                else:
                    self.distance_to_non_collisions[bucket_num] = self.distance_to_non_collisions[bucket_num] + 1


    def plot(self, requested_label):
        r = self.getPointsBound()
        interval_size = (2*r)/float(self.num_of_distance_buckets)
        plt.plot(
            [(bucket_num*interval_size)/float(1) for bucket_num in self.distance_to_collisions.keys()], \
            [self.distance_to_collisions[bucket_num]/float(self.distance_to_collisions[bucket_num] + self.distance_to_non_collisions[bucket_num]) for bucket_num in self.distance_to_collisions.keys()], 'o')
        plt.plot(
            [(bucket_num*interval_size)/float(1) for bucket_num in self.distance_to_collisions.keys()], \
            [self.distance_to_collisions[bucket_num]/float(self.distance_to_collisions[bucket_num] + self.distance_to_non_collisions[bucket_num]) for bucket_num in self.distance_to_collisions.keys()], \
            '--', label=requested_label)

    def plot_normalize(self, requested_label, c):
        r = self.getPointsBound()
        interval_size = (2*r)/float(self.num_of_distance_buckets)
        plt.plot(
            [(bucket_num*interval_size)/float(c) for bucket_num in self.distance_to_collisions.keys()], \
            [self.distance_to_collisions[bucket_num]/float(self.distance_to_collisions[bucket_num] + self.distance_to_non_collisions[bucket_num]) for bucket_num in self.distance_to_collisions.keys()], 'o')
        plt.plot(
            [(bucket_num*interval_size)/float(c) for bucket_num in self.distance_to_collisions.keys()], \
            [self.distance_to_collisions[bucket_num]/float(self.distance_to_collisions[bucket_num] + self.distance_to_non_collisions[bucket_num]) for bucket_num in self.distance_to_collisions.keys()], \
            '--', label=requested_label)

class GausssianPartitioning(LSH):
    def __init__(self, eta, c, d):
        LSH.__init__(self, d)
        self.init(eta, c, d)

    def init(self, eta, c, d):
        self.epsilon = d**(-0.5)
        self.eta = eta
        self.c = c
        self.d = d
        self.P = []
        #self.jl = JLGausssianMatrix(0.1, d, 250)
        #self.d = self.jl.getMinK()

        #copy points
        for i in range(2**self.d):
            w = np.random.randn(self.d)
            self.P.append(w)
        print 'Done building'
    #incrementatly change c
    def set_c(self, c):
        self.c = c

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def h(self, p):
        p = (p / np.linalg.norm(p))*self.eta*self.c;
        i = 0
        for w in self.P:
            if np.dot(p, w) >= self.eta*self.c*self.epsilon*(self.d**0.5):
                return i
            i = i + 1

        return None

    def getPointsBound(self):
        return self.eta*self.c

    def norm(self, p):
        return np.linalg.norm(p)



        
class HammingLSH:
    #points live in {0,..,M}^d
    def __init__(self, d):
        self.d = d
        self.i = np.random.randint(0, d)
    def h(self, p):
        return p[i]


class L1LSH(LSH):
    #d - the dim
    #delta - approximation factor
    #a - points live in [0,a]^d
    #number of functions to concacate
    def __init__(self, d, delta, a, k):
        LSH.__init__(self, d)

        self.a = a
        self.M = (a*d)/float(delta)
        self.delta = delta
        self.s = float(self.d)/self.delta
        self.k = k

        self.selected_dim = []
        self.offset_in_dim = []

        for j in range(k):
            i = np.random.randint(0, d*self.M)
            self.selected_dim.append(i / self.M)
            self.offset_in_dim.append(i % self.M)


    def h(self, p):
        
        res = []
        for j in range(len(self.selected_dim)):
            selection = np.around(p*self.s)[self.selected_dim[j]]
            if selection <= self.offset_in_dim[j]: #if selection is smaller than self.offset_in_dim, it means there is a one in its dM unary representation
                res.append(1)
            else:
                res.append(0)
        return res

    def getPointsBound(self):
        return self.a

    def norm(self, p):
        return np.linalg.norm(p, 1)



#(self, d, delta, a, k):
"""
lsh = L1LSH(20, 0.5, 100.0, 5)
lsh.test(500)
lsh.plot('k = ' + str(lsh.k))

lsh = L1LSH(20, 0.5, 100.0, 10)
lsh.test(500)
lsh.plot('k = ' + str(lsh.k))

lsh = L1LSH(20, 0.5, 100.0, 15)
lsh.test(500)
lsh.plot('k = ' + str(lsh.k))


plt.ylabel('Collisions ratio')
plt.xlabel('Distance')
plt.legend()
plt.show()
"""


lsh = GausssianPartitioning(1, 1, 20)
orig_eps = lsh.c

lsh.set_c(0.5*orig_eps)
lsh.test(500)
lsh.plot_normalize('c = ' + str(lsh.c), lsh.c)

lsh.set_c(1*orig_eps)
lsh.test(500)
lsh.plot_normalize('c = ' + str(lsh.c), lsh.c)

lsh.set_c(1.5*orig_eps)
lsh.test(500)
lsh.plot_normalize('c = ' + str(lsh.c), lsh.c)

plt.ylabel('Collisions ratio')
plt.xlabel('Distance/c')
plt.legend()
plt.show()


"""
n = 250

par = GausssianPartitioning(1, 1, 20)
orig_eps = par.epsilon

par.set_epsilon(0.5*orig_eps)
par.test(n)
par.plot_epsilon(True)

par.set_epsilon(orig_eps)
par.test(n)
par.plot_epsilon(True)

par.set_epsilon(orig_eps*2)
par.test(n)
par.plot_epsilon(True)

plt.ylabel('Collisions ratio')
plt.xlabel('Distance / c')
plt.legend()
plt.show()




par.set_epsilon(0.5*orig_eps)
par.test(n)
par.plot_epsilon(False)

par.set_epsilon(orig_eps)
par.test(n)
par.plot_epsilon(False)

par.set_epsilon(orig_eps*2)
par.test(n)
par.plot_epsilon(False)

plt.ylabel('Collisions ratio')
plt.xlabel('Distance')
plt.legend()
plt.show()
"""