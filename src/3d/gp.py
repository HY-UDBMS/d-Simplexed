from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np

# f1,f2
X = [
    [1,1],
    [10,    5],
    [20 ,   5],
    [30,    5],
    [40,    5],
    [50,    5],
    [60,    5],

    [10,    10],
    [20,    10],
    [30,    10],
    [40,    10],
    [50,    10],
    [60,    10],

    [10,    15],
    [20,    15],
    [30,    15],
    [40,    15],
    [50,    15],
    [60 ,   15],

    [10 ,   20],
    [20 ,   20],
    [30  ,  20],
    [40 ,   20],
    [50 ,   20],
    [60,    20],

    [10 ,   25],
    [20,    25],
    [30,    25],
    [40 ,   25],
    [50,    25],
    [60,    25],

    [60,	45],
    [60,	60]


]

# runtime(f1,f2)
Y = [
    10,

    6.3687,
    4.33018,
    1.49257,
    1.38429,
    1.37552,
    1.35659,

    5.18841,
    2.87083,
    1.08855,
    1.10772,
    1.01886,
    1.00626,

    5.15312,
    2.66598,
    0.93767,
    0.90635,
    1.06719,
    0.97266,

    3.26359,
    2.2035,
    0.85163,
    0.93339,
    0.91893,
    0.90539,

    3.53785,
    2.13402,
    0.94102,
    0.89912,
    0.88065,
    0.87362,

    0.86362,
    0.86362
]

# Instanciate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-5, 1e5))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, Y)

testX = [(60, x) for x in np.linspace(1, 70, 1000)]

#print(testX)
# # Make the prediction on the meshed x-axis (ask for MSE as well)
#y_pred, sigma = gp.predict(testX, return_std=True)

y_pred, sigma = gp.predict([[15,8]], return_std=True)

#gp.predict()

print str(y_pred)

