import Bootstrap
import numpy as np
from scipy import stats

# This example (Section 8.3, page 88, "Mouse Data" from text)
#
# This exercise is "a pair of probability distributions F and G,
# the first for the Treatment group and the second for the Control group
#
#       P = (F, G)
#
# I did not complete this exercise.
# To be done:
# Update the Boostrap library (or create a new version of it) that
# could handle multi-distribution case.  This is a common case so
# certainly would be worth the effort.
#
# Specifically: the "s" function would take in (treatment, control) and calculate
# the measure (which in this case is the difference of the means) and return that
# The bootstrap would take in two sets of original data (treatment, control) and
# create the x_star_treatment, x_start_control and send those to the "s" function.

class TestClass(object):

    # "s" (the measure) in this case is (intended to be)the difference between
    # the means of treatment and control.  Code below is incorrect.
    def my_s(self, x_star_i):

        return( np.mean(x_star_i))


    def test_treatment(self):

        treatment = np.array([94, 197, 16, 38, 99, 141, 23])
        s = self.my_s
        B = 100

        # look at the original data...
        print("Treatment sample size: ", treatment.shape[0], " mean: ", np.mean(treatment), "sem: ", stats.sem(treatment))

        # run the bootstrap
        ### this is incorrect - we actually should run boot strap
        # on the DIFFERENCE btween Treatment and control
        bootstrap = Bootstrap.Bootstrap(treatment, s, B)
        [std, sem] = bootstrap.run()


    def test_control(self):
        # again incorrect - the control and treatment are both in a single exercise - not separate
        control = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46])
