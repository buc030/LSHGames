import math
import numpy as np
import matplotlib.pyplot as plt
import itertools

#dim reduction d -> k
class JLGausssianMatrix:
    def __init__(self, epsilon, d, k, n):
        self.epsilon = epsilon
        self.n = n
        self.d = d
        self.k = k

        assert k >= self.getMinK()
        #construct the matrix:
        sigma = (1.0/k)**0.5
        self.matrix = sigma*np.random.randn(k, d)

    def getMinK(self):
        return math.ceil((4*( (self.epsilon**2)/2.0 - (self.epsilon**3)/3.0)**-1)*math.log(self.n))

    def map(self, x):
        return np.dot(self.matrix, x)


class GausssianPartitioning:
    def __init__(self, eta, c, d, epsilon):
        self.init(eta, c, d, epsilon)

    def init(self, eta, c, d, epsilon):
        self.epsilon = epsilon
        self.eta = eta
        self.c = c
        self.d = d
        self.P = []
        #copy points
        for i in range(2**d):
            w = np.random.randn(self.d)
            self.P.append(w)

    
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
        p = np.random.randn(self.d - 1)
        p = p / np.linalg.norm(p)
        p = p*np.random.random() 
        #print 'r = ' + str(r)
        #print 'norm = ' + str(np.linalg.norm(p))
        p = np.append(p, (r**2 - np.linalg.norm(p)**2)**0.5)
        return p

    def test(self, num_test_points):
        num_of_distance_buckets = 100
        r = self.eta*self.c
        distance_to_collisions = {}  
        distance_to_non_collisions = {}  
        #max distance is 2cr, distance buckets:
        #(0 - 2cr/num_of_distance_buckets) , (2cr/num_of_distance_buckets, 4cr/num_of_distance_buckets) 
        #(i2cr/num_of_distance_buckets, (i+1)2cr/num_of_distance_buckets) 
        interval_size = (2*r)/float(num_of_distance_buckets)

        #the point must be on a sphere:
        points = [self.generateRandomPointOnSphere() for i in range(num_test_points)]
        
        for (p1, p2) in itertools.combinations(points, 2):
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
            [bucket_num*interval_size for bucket_num in distance_to_collisions.keys()], \
            [distance_to_collisions[bucket_num]/float(distance_to_collisions[bucket_num] + distance_to_non_collisions[bucket_num]) for bucket_num in distance_to_collisions.keys()], 'o')
        plt.plot(
            [bucket_num*interval_size for bucket_num in distance_to_collisions.keys()], \
            [distance_to_collisions[bucket_num]/float(distance_to_collisions[bucket_num] + distance_to_non_collisions[bucket_num]) for bucket_num in distance_to_collisions.keys()], \
            '--', label='Num of collisons / Total num of pairs within distance')

        """
        plt.plot(
            [bucket_num*interval_size for bucket_num in distance_to_collisions.keys()], \
            [distance_to_collisions[bucket_num] for bucket_num in distance_to_collisions.keys()], 'o')
        plt.plot(
            [bucket_num*interval_size for bucket_num in distance_to_collisions.keys()], \
            [distance_to_collisions[bucket_num] for bucket_num in distance_to_collisions.keys()], '--', label='Num of Collisions')


        plt.plot(
            [bucket_num*interval_size for bucket_num in distance_to_non_collisions.keys()], \
            [distance_to_non_collisions[bucket_num] for bucket_num in distance_to_non_collisions.keys()], 'o')
        plt.plot(
            [bucket_num*interval_size for bucket_num in distance_to_non_collisions.keys()], \
            [distance_to_non_collisions[bucket_num] for bucket_num in distance_to_non_collisions.keys()], '--', label='Num of Non Collisions')
        """
        
        plt.ylabel('Collisions ratio')
        plt.xlabel('Distance')
        plt.legend()
        plt.show()
        print distance_to_collisions


par = GausssianPartitioning(1, 1.5, 20, 0.1)
par.test(1000)