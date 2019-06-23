import Bootstrap
import numpy as np

# Some basic tests on my bootstrap "library" which is a single file here in
# the project /lib/Bootstrap.py


class TestClass(object):

    def test_single_variate(self):
        # this is a single-variable expressed as a 2-d numpy array
        # (the attribute is column 0, rows are instances)
        num_instances = 100
        num_attributes = 1

        X = np.random.randint(5, size=(num_instances, num_attributes))
        s = np.mean
        B = 2

        bootstrap = Bootstrap.Bootstrap(X,s,B)
        bootstrap.run()

        assert bootstrap.N == num_instances
        assert bootstrap.B == B

    def test_single_variate_single_dimension(self):
        # this is a single-variable expressed as a 1-d numpy array
        # (each element of the array is an instance)
        X = np.array([1, 2, 3, 4, 5])
        B = 2
        s = np.mean
    
        bootstrap = Bootstrap.Bootstrap(X, s, B)
        bootstrap.run()

        assert bootstrap.N == 5
        assert bootstrap.B == B


    def test_multi_variate(self):
        # this is a three-variable expressed as a 2-d numpy array
        # (the attributes are columns, rows are instances)
        num_instances = 6
        num_attributes = 3

        X = np.random.randint(5, size=(num_instances, num_attributes))
        s = np.mean
        B = 2

        bootstrap = Bootstrap.Bootstrap(X,s,B)
        bootstrap.run()

        assert bootstrap.N == num_instances
        assert bootstrap.B == B

