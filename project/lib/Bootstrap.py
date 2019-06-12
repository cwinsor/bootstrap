# This is the Bootstrap class

import numpy as np
from scipy import stats

class Bootstrap(object):

    def __init__(self, X, s, B):

        # Input parameters:
        # X = empirical sample in format of instances, not percentages. For multi-variate
        #   distributions it is assumed the first index the sample number, the second
        #   index is the attribute number
        # s = function which computes the statistic
        # B = number of bootstrap iterations desired

        # N is the number of samples in X
        self.N = X.shape[0]

        # if single-variate data is send in as a single-dimensional array
        # convert it to two dimensions where the second dimension is "1" (the attribute)
        if len(X.shape)==1:
            self.X = X.reshape(self.N,1)
        else:
            self.X = X
        self.s = s
        self.B = B

    def run(self, callback=None):

        # Input parameter:
        #   callback:  this is a caller-provided function that is called after
        #              the bootstrap sample is created and theta_star calculated
        #              it allows the user to do whatever they want with the data

        # we are collecting theta_star[b] - this is the measure
        # computed by the "s" function on the empirical sample x_star[b]
        self.theta_star = np.zeros((self.B))

        # loop B times
        # the elements of x_star[b] are sampled with replacement from X
        for b in range(self.B):
            idx = np.random.randint(self.N, size=self.N)
            x_star = self.X[idx,:]

            # perform the "s" function using the x_star[b] data
            # capturing the result theta_star[b]
            self.theta_star[b] = self.s(x_star)

            # perform the callback - if the caller wants to do anything more with
            # the data this is their chance
            if callback != None:
                callback(x_star, self.theta_star[b])

        # return [standard deviation, standard error] of theta_star[]
        return [np.std(self.theta_star), stats.sem(self.theta_star)]

