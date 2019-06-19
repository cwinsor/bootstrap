import Bootstrap
import numpy as np
import data_chapter_07_a_eigen as my_data
import my_charts
import pytest

# This test calculates covariance by hand
# and compares to results from numpy

def test_07b_calculate_covariance_by_hand():
        from numpy import linalg as LA

        # calculate by hand
        X = my_data.get_data()
        [n,m] = X.shape
        n_float = np.float64(n)
        x_bar = np.sum(X, axis=0) / n_float
        print("x_bar ", x_bar)

        g = np.zeros((m,m))
        for j in range(m):
                for k in range(m):
                        sum = 0.0
                        for i in range(n):
                                term1 = X[i,j] - x_bar[j]
                                term2 = X[i,k] - x_bar[k]
                                product = term1 * term2
                                sum = sum + product
                                #print(j, " ", k, " ", sum)
                        g[j,k] = sum
        g = g / n

        print("--- by hand ---\n",g)


        # calculate using numpy
        covariance_matrix = np.cov(X, bias=True, rowvar=False)
        print("--- using numpy ---\n",covariance_matrix)

        (m,n) = g.shape
        for row in range(m):
                for col in range(n):
                        first = g[row,col]
                        second = covariance_matrix[row,col]
                        print(first, "  ", second)
                        assert(first == pytest.approx(second, rel=None, abs=1e-7))

        #assert(False)

        



