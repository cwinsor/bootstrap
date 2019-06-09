import Bootstrap
import numpy as np

def test_single_variate():
    num_instances = 6
    num_attributes = 1

    X = np.random.randint(5, size=(num_instances, num_attributes))
    s = np.mean
    B = 2

    bootstrap = Bootstrap.Bootstrap(X,s,B)
    bootstrap.run()

    assert bootstrap.N == num_instances
    assert bootstrap.B == B
    assert bootstrap.mean() == np.mean(X)


def test_single_variate_single_dimension():

    X = np.array([1, 2, 3, 4, 5])
    B = 2
    s = np.mean
  
    bootstrap = Bootstrap.Bootstrap(X, s, B)
    bootstrap.run()

    assert bootstrap.N == 5
    assert bootstrap.B == B
    assert bootstrap.mean() == np.mean(X)

def test_multi_variate():
    num_instances = 6
    num_attributes = 3

    X = np.random.randint(5, size=(num_instances, num_attributes))
    s = np.mean
    B = 2

    bootstrap = Bootstrap.Bootstrap(X,s,B)
    bootstrap.run()

    assert bootstrap.N == num_instances
    assert bootstrap.B == B
    assert bootstrap.mean() == np.mean(X)

