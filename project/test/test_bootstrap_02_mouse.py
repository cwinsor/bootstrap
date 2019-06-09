import Bootstrap
import numpy as np

def test_treatment():

    treatment = np.array([94, 197, 16, 38, 99, 141, 23])
    s = np.mean
    B = 2

    bootstrap = Bootstrap.Bootstrap(treatment, s, B)
    bootstrap.run()

    assert bootstrap.N == 7
    assert bootstrap.B == B
    assert bootstrap.mean() == np.mean(treatment)


def test_control():
    control = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46])
    s = np.mean
    B = 2

    bootstrap = Bootstrap.Bootstrap(control, s, B)
    bootstrap.run()

    assert bootstrap.N == 9
    assert bootstrap.B == B
    assert bootstrap.mean() == np.mean(control)
