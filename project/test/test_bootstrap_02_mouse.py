import Bootstrap
import numpy as np
from scipy import stats

# This example (Chapter 2, Table 2.1, page 11 "Mouse Data" in text)
# This just runs the Bootstrap procedure on the two tables.

class TestClass(object):

    # "s" (the measure) in this case is just the sample mean
    def my_s(self, x_star_i):

        return( np.mean(x_star_i))


    def test_treatment(self):

        treatment = np.array([94, 197, 16, 38, 99, 141, 23])
        s = self.my_s
        B = 100

        # look at the original data...
        print("Treatment sample size: ", treatment.shape[0], " mean: ", np.mean(treatment), "sem: ", stats.sem(treatment))

        # run the bootstrap
        ### this is incorrect - we actually should run boot strap on the DIFFERENCE btween Treatment and control - see 
        bootstrap = Bootstrap.Bootstrap(treatment, s, B)
        [std, sem] = bootstrap.run()

        print("Treatment Bootstrap standard deviation of mean: ",std, "standard error of the mean", sem)
        #assert(False)


    def test_control(self):
        control = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46])
        s = np.mean
        B = 2