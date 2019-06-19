import Bootstrap
import numpy as np
import data_chapter_07_a_eigen as my_data
import my_charts

# This example (Section 7.2 Example 1, "test score data" on page 62 in text)
# calculates standard error on a measure relating
# to eigenvectors and eigenvalues.
#
# The dataset is from Mardia, Kent and Bibby (1979), 88 samples of student test
# scores on five tests (mechanics,
# vectors, algebra, analysis and statistics).  There is a high level of
# correlation in the data - a student that does well in one subject frequently
# does well in other subjects. This correlation between attributes is reflected
# in the covariance matrix. A covariance matrix is a diagonal matrix
# indicating the degree of correlation between variable x and variable y. The
# measure of correlation is the sum of the product of (the difference between the ith
# element m and its mean, times the difference between the ith variable n and its mean).
# In this case there are 5 variables so the covariance matrix is a 5x5.
#
# Eigenvalues and eigenvectors of the covariance matrix give insight into the relationship
# between the variables.
# Eigenvectors are the 'principle components' of the matrix under investigation (in this case the
# covariance matrix).
# A 5x5 covariance matrix has 5 eigenvectors and 5 eigenvalues. Each eigenvector
# is the direction which minimizes the sum-of-the-squared-distances
# in that direction.
# The first vector attempts to minimize this measure without any constraint
# The second (and all subsequent) is/are orthogonal to to the predecessors.
# The eigenvalue, of which there are 5 in this case, gives the size of the vector whi
# is also an indication of the correlation between the data and the vector.
#
# In this example (page 62) the "s" (measure we are interested in) is a ratio of the
# first eigenvalue to the sum of the eigenvalues.  What this ratio expresses is
# how significant the first eigenvector is compared to all the others. 
#
# The example in the text - if the correlation between samples were perfect then
# a relationship could be expressed as
#   x(i) = Q(i)v
# where
#    v is the first eigenvector
#    Q(i) is a scale factor for the ith student
#    x(i) is the scores for the ith student
# This is obvioiusly not the case, but it serves to illustrate what a
# strong eigenvector_1 would be.
#
# In the code below - the "s" is:
#   compute the 5 eigenvalues:   eigenvalues[5]
#   compute the ratio:  eigenvalues[1] / sum(eigenvalues)
#
# So the function below implements "s".
# It receives x_star_i (the bootstrap sample),
# it computes the covariance matrix (5x5),
# then from that the eigenvalues,
# then the ratio of first eigenvalue to the sum of eigenvalues

def ratio_first_eigenvector_to_sum(x_star_i):
    from numpy import linalg as LA

    covariance_matrix = np.cov(x_star_i, bias=True)
    w, v = LA.eig(covariance_matrix)
    v = np.transpose(v)

    return( w[0] / sum(w) )

# global variables used as scratchpad to keep results
# the arrays are resized in the test once the data size is known
#eigenvalues = >>> a = numpy.zeros(shape=(5,2))
#eigenvectors = np.array([])
#eigenvalues = np.array([])


def test_07a():
    from numpy import linalg as LA

    #eigenvectors = np.array([])
    #eigenvalues = np.array([])

    X = my_data.get_data()
    print(X)
    s = ratio_first_eigenvector_to_sum
    #B = 200
    B = 3

    # explore the empirical data...
    covariance_matrix = np.cov(X, bias=True, rowvar=False)
    w, v = LA.eig(covariance_matrix)
    v = np.transpose(v)
    print("--- empirical data - shape ---")
    print(X.shape)
    print("--- empirical data - covariance matrix ---")
    print(covariance_matrix)
    print("--- empirical data - eigenvalues ---")
    print(w)
    print("--- empirical data - eigenvectors ---")
    print(v)

    # run the bootstrap
    print("--- run the bootstrap ---")
    bootstrap = Bootstrap.Bootstrap(X,s,B)
    bootstrap.add_callback(my_callback)
    [std, sem] = bootstrap.run()

    print("standard deviation:")
    print(std)
    print("standard error of the mean")
    print(sem)
    #assert(False)

    # investigate the results
    # plot the theta_stars (the measure) from the bootstrap replications
    # the expectation is this is somewhat gaussian (long tails are not acceptable)
    print("--- results from bootstrap ---")
    #print(bootstrap.theta_star)
    my_charts.plot_histogram(bootstrap.theta_star, "Count of Occurrences", "Ratio: eigenV1/sum(eigen)", "Histogram - Count of EigenV1/sum")
    assert(False)

    # plot the first two principal component vectors using box-and-whisker
    #print("eigenvalues\n", eigenvalues)
    #print("eigenvectors\n", eigenvectors)
    assert(False)



def my_callback(x_star, theta_star_b):
        from numpy import linalg as LA

        # in the callback we capture principal component vectors (eigenvectors)
        covariance_matrix = np.cov(np.transpose(x_star), bias=True)
        w, v = LA.eig(covariance_matrix)
        v = np.transpose(v)
        print("here")

        eigenvectors = np.concatenate(eigenvectors, v)
        eigenvalues = np.concatenate(eigenvalues, w)
