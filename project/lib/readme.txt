
======= actual population (to which we do not have access) =======
F = actual population
t() a function which performs a measure of which we are of interested (e.g. mean, correlation, eigen)
theta = the value of the measure on the population, i.e. t(F)

======= empirical sample =========
X = empirical sample
N = number of samples in X

======== bootstrap =========
x_star[b] = bootstrap sample (sample number b, of size N taken from X with replacement)
s() = the "statistic" function
B = the number of bootstrap replicates desired (typically 50-200)
s[b] = s(x_star[b]) = "bootstrap replicate" b - this is the value resulting from running s() on x_star[b]
se_boot = standard error or bootstrap replicate measures

<question - does s() need to be the same as t() ?  I would think so.

+=====================
implementation (pseudo-code) ...

function (X, s, B)
where
X = empirical sample in format of instances, not percentages. For multi-variate
    distributions it is assumed the first index the sample number, the second
    index is the attribute number
s = function which computes the statistic
B = number of bootstrap iterations desired

//N = sizeof(X) // the size of the empirical sample - used to create the 
x_star = new array of size [B] where each element same shape as x

foreach x_star[b]:
    x_star[b] = random_sample_with_replacement(x)
    s_star[b] = s(x_star[b])

se_boot = standard_error(s[])
sd_boot = standard_deviation(s[])






theta_star[n] = measure of theta on f_bar
x = 
model population
f_bar_hat = 