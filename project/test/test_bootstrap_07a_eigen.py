import Bootstrap
import numpy as np

# This example (page 62 in text) calculates standard error on a measure relating
# to eigenvectors and eigenvalues.
#
# The dataset is 88 samples of student test scores on five tests (mechanics,
# vectors, algebra, analysis and statistics).  There is a high level of
# correlation in the data - a student that does well in one subject frequently
# does well in other subjects. This correlation between samples expresses
# itself in the covariance matrix. A covariance matrix is a diagonal matrix
# indicating the degree of correlation between variable x and variable y. The
# measure of correlation is the sum of the product of (the difference between the ith
# element m and its mean, times the difference between the ith variable n and its mean).
# In this case the covariance matrix is a 5x5.
#
# Eigenvalues and eigenvectors of the covariance matrix give insight into the relationship
# between the variables.
# Eigenvectors are the 'principle components' of the target matrix (in this case the
# covariance matrix).
# A 5x5 covariance matrix has 5 eigenvectors and 5 eigenvalues. The vector (the eigenvector)
# is the direction which minimizes the sum-of-the-squared-distances. 
# The first vector attempts to minimize this measure, the second is orthogonal to
# the first and attempts to minimize the remaining sum of squared-distances, etc.
# The eigenvalue gives the size of the vector.
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

    covariance_matrix = np.cov(x_star_i)
    w, v = LA.eig(covariance_matrix)
    return( w[0] / sum(w) )

# global variables - scratchpad to keep results
# this is resized in the test once the data size is known
#eigenvalues = >>> a = numpy.zeros(shape=(5,2))
#eigenvectors = []
#eigenvalues = []

def test_07a():
    from numpy import linalg as LA

    X = get_data()
    s = ratio_first_eigenvector_to_sum
    B = 200

    # explore the empirical data...
    print("--- empirical data ---")
    print(X.shape)
    covariance_matrix = np.cov(np.transpose(X))
    w, v = LA.eig(covariance_matrix)
    print(covariance_matrix)
    print(v)
    print(w)
    #assert(False)

    # run the bootstrap
    print("--- run the bootstrap ---")
    bootstrap = Bootstrap.Bootstrap(X,s,B)
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
    plot_histogram(bootstrap.theta_star, "Count of Occurrences", "Ratio: eigenV1/sum(eigen)", "Histogram - Count of EigenV1/sum")
    assert(False)

    # plot the first two principal component vectors using box-and-whisker
    # we 


    def my_callback(x_star, theta_star_b):

        # in the callback we capture principal component vectors (eigenvectors)
        covariance_matrix = np.cov(np.transpose(x_star))
        w, v = LA.eig(covariance_matrix)

        print("here")



def get_data():

    # MC  VC  LO  NO  SO
    s = '''\
77  82  67  67  81
63  78  80  70  81
75  73  71  66  81
55  72  63  70  68
63  63  65  70  63
53  61  72  64  73
51  67  65  65  68
59  70  68  62  56
62  60  58  62  70
64  72  60  62  45
52  64  60  63  54
55  67  59  62  44
50  50  64  55  63
65  63  58  56  37
31  55  60  57  73
60  64  56  54  40
44  69  53  53  53
42  69  61  55  45
62  46  61  57  45
31  49  62  63  62
44  61  52  62  46
49  41  61  49  64
12  58  61  63  67
49  53  49  62  47
54  49  56  47  53
54  53  46  59  44
44  56  55  61  36
18  44  50  57  81
46  52  65  50  35
32  45  49  57  64
30  69  50  52  45
46  49  53  59  37
40  27  54  61  61
31  42  48  54  68
36  59  51  45  51
56  40  56  54  35
46  56  57  49  32
45  42  55  56  40
42  60  54  49  33
40  63  53  54  25
23  55  59  53  44
48  48  49  51  37
41  63  49  46  34
46  52  53  41  40
46  61  46  38  41
40  57  51  52  31
49  49  45  48  39
22  58  53  56  41
35  60  47  54  33
48  56  49  42  32
31  57  50  54  34
17  53  57  43  51
49  57  47  39  26
59  50  47  15  46
37  56  49  28  45
40  43  48  21  61
35  35  41  51  50
38  44  54  47  24
43  43  38  34  49
39  46  46  32  43
62  44  36  22  42
48  38  41  44  33
34  42  50  47  29
18  51  40  56  30
35  36  46  48  29
59  53  37  22  19
41  41  43  30  33
31  52  37  27  40
17  51  52  35  31
34  30  50  47  36
46  40  47  29  17
10  46  36  47  39
46  37  45  15  30
30  34  43  46  18
13  51  50  25  31
49  50  38  23  09
18  32  31  45  40
08  42  48  26  40
23  38  36  48  15
30  24  43  33  25
03  09  51  47  40
07  51  43  17  22
15  40  43  23  18
15  38  39  28  17
05  30  44  36  18
12  30  32  35  21
05  26  15  20  20
00  40  21  09  14
    '''
    s_array = np.fromstring(s, dtype=int, sep=' ')
    s_array2 = np.reshape(s_array, (-1, 5))
    return s_array2






def plot_histogram(x, ylabel, xlabel, title):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # the histogram of the data
    n, bins, patches = plt.hist(x, facecolor='g')
    ymin = 0
    ymax = max(n) * 1.1
    xmin = min(bins) * 0.9
    xmax = max(bins) * 1.1

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.text(60, .025, r'foobar')
    plt.axis([xmin, xmax, ymin, ymax])
    plt.grid(True)
    plt.show()

