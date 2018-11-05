from scipy.spatial import Delaunay
import numpy as np
import random
import itertools

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import LinearRegression

#input array, in 2-d it is points
def build_triangulation(array):
    tri = Delaunay(array)
    return tri


#input list X parameters, Y runing time
def build_gp_model(X,Y):
    # Instanciate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-5, 1e5))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, Y)
    return gp


#input list X parameters, Y running time
def build_lr_model(X,Y):
    lr = LinearRegression().fit(X, Y)
    return lr

#find the simplex
def find_simplex(tri, array_data, test):
    return array_data[tri.simplices[tri.find_simplex(test)]]


def read_data(filename):
    f = open(filename, 'r')
    header = f.readline()
    res = f.readlines()
    f.close()
    return [x.rstrip().split(' ') for x in res]

#from simplex to linear equation
def solve_linear(parameters, T):
    # a = np.array([[2, -4, 4], [34, 3, -1], [1, 1, 1]])
    # b = np.array([8, 30, 108])
    a = np.array(parameters)
    b = np.array(T)
    x = np.linalg.solve(a, b)
    return x

#prediction from simplex
def prediction(array_data,simplex,test):
    # compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
    num_features = 3
    simplex_result =[]
    for s in simplex:
        for k in array_data:
            if list(s)==k[:num_features]:
                simplex_result.append(k)
                break

    parameters = [[int(x[0]),int(x[1]),int(x[2]),1] for x in simplex_result]
    T = [float(x[3]) for x in simplex_result]
    # print(parameters)
    # print(T)
    # print(simplex_result)
    # print(solve_linear(parameters, T))
    # print('test')
    # print(test[:3])
    # print('result')
    # print(np.dot(solve_linear(parameters, T),test))
    return list(test)[:3]+[np.dot(solve_linear(parameters, T),test)]

#get discrete lhs samples
def discrete_lhs(feature_space, sample_size):
    #from boundary to all seeds
    x = [list(range(x[0],x[1]+1,sample_size)) for x in feature_space]
    #shuffle the seeds
    for l in x:
        random.shuffle(l)
    result = []
    #get combinations
    while len(x[0])>0:
        tmp =[]
        for l in x:
            tmp.append(l.pop())
        result.append(tmp)
    return result

#get the min max boundary points
def boundary(feature_space):
    result=list(itertools.product(*feature_space))
    return [list(x) for x in result]

def precision(tri, samples, samples_with_t, test_with_t):
    preci = []
    for t in test_with_t:
        simplex = find_simplex(tri, np.array(samples), t[:3])
        num_features = 3
        simplex_result = []
        for s in simplex:
            for k in samples_with_t:
                if list(s) == k[:num_features]:
                    simplex_result.append(k)
                    break

        parameters = [[int(x[0]), int(x[1]), int(x[2]), 1] for x in simplex_result]
        T = [float(x[3]) for x in simplex_result]

        # |(Ti-Ti')|/Ti
        preci.append(abs(t[3] - np.dot(solve_linear(parameters, T), t[:3]+[1]))/t[3])
    # print(preci)
    return sum(preci)/len(preci) #avg


def precision_gp(gp, samples_with_t, test_with_t):

    T = [x[3] for x in test_with_t]
    y_pre, sigma = gp.predict([x[:3] for x in test_with_t], return_std=True)
    # print(list(y_pre))
    # print(T)
    preci = list(abs(T-y_pre)/T)
    # print(preci)
    return sum(preci) / len(preci)  # avg

def precision_lr(lr, samples_with_t, test_with_t):

    T = [x[3] for x in test_with_t]
    y_pre = lr.predict([x[:3] for x in test_with_t])
    # print(list(y_pre))
    # print(T)
    preci = list(abs(T-y_pre)/T)
    # print(preci)
    return sum(preci) / len(preci)  # avg

def prediction_for_samples(tri, samples, samples_with_t, tests):
    new_tests=[]
    for t in tests:
        simplex = find_simplex(tri, np.array(samples), t)
        num_features = 3
        simplex_result = []
        for s in simplex:
            for k in samples_with_t:
                if list(s) == k[:num_features]:
                    simplex_result.append(k)
                    break
        parameters = [[int(x[0]), int(x[1]), int(x[2]), 1] for x in simplex_result]
        T = [float(x[3]) for x in simplex_result]
        new_tests.append(t+[np.dot(solve_linear(parameters, T), t+[1])])
    return new_tests

def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5

def utility(tri, samples, samples_with_t, new_samples):
    dist = []
    max_dist = 0

    #normalize to 1
    new_samples_normed = [list(x) for x in np.array(new_samples) / np.array(new_samples).max(axis=0)]

    for i in range(len(new_samples)):
        # print(new_samples[i])
        simplex = find_simplex(tri, np.array(samples), new_samples[i][:3])
        tmp_dist = 0
        simplex_with_t = [x for x in samples_with_t if x[:3]in simplex]
        simplex_with_t_normed = [list(x) for x in np.array(simplex_with_t) / np.array(simplex_with_t).max(axis=0)]
        for j in range(len(simplex)):
            tmp_dist = tmp_dist + euclidean(new_samples_normed[i], simplex_with_t_normed[j])

        dist.append(new_samples[i]+[tmp_dist/len(simplex)])
    # print('aaa')
    # print(dist)
    # sorted(dist, key=lambda x: x[4])
    # print()
    return sorted(dist, key=lambda x: -x[4])

#mian driver
def main_driver(feature_space, ground_data):
    sample_size = 1
    #initial seeds
    samples = boundary(feature_space)
    print('initial size: '+str(len(samples)))
    num_parameters = 3
    samples_with_t = [x for x in ground_data if x[:num_parameters] in samples]

    #parameter array
    parameter_data = [x[:num_parameters] for x in ground_data if x[0]]

    #test set
    test = random.sample(parameter_data, int(0.2*len(parameter_data)))
    test_with_t = [x for x in ground_data if x[:num_parameters] in test]

    #initial triangulation model
    tri = build_triangulation(samples)

    #initial linear model
    lr = build_lr_model([x[:3] for x in samples_with_t], [x[3] for x in samples_with_t])

    #initial gp model
    gp = build_gp_model([x[:3] for x in samples_with_t], [x[3] for x in samples_with_t])


    cnt = 1
    rounds = 51

    filename = 'data/kmean_result.dat'
    with open(filename, "a") as f:

        while cnt <= rounds:

            #precision for tri
            preci = precision(tri, samples, samples_with_t, test_with_t)
            print('rounds: '+str(cnt))
            print('tri precision: ' + str(preci))

            preci_gp = precision_gp(gp, samples_with_t, test_with_t)
            print('rounds: '+str(cnt))
            print('gp precision: ' + str(preci_gp))

            preci_lr = precision_lr(lr, samples_with_t, test_with_t)
            print('rounds: '+str(cnt))
            print('lr precision: ' + str(preci_lr))


            f.write(str(8+10*(cnt-1)) + ' ' + str(round(preci*100, 3)) + ' ' +  str(round(preci_gp*100,3)) + ' ' + str(round(preci_lr*100,3)) + "\n")  # quantity incategory

            new_sample_dist = []
            while new_sample_dist == []:
                new_samples = [x for x in discrete_lhs(feature_space, sample_size) if x not in test and x not in samples]
                # print(new_samples)
                if new_samples != []:
                    new_samples_t_ = prediction_for_samples(tri, samples, samples_with_t, new_samples)
                    new_sample_dist = utility(tri, samples, samples_with_t, new_samples_t_)

            # add k of lhs samples
            k=10
            samples = samples +[x[:3] for x in new_sample_dist[:k]]
            samples_with_t = samples_with_t + [x for x in ground_data if x[:3] in [x[:3] for x in new_sample_dist[:k]]]

            #update model todo in a incremental way
            tri = build_triangulation(samples)
            gp = build_gp_model([x[:3] for x in samples_with_t], [x[3] for x in samples_with_t])
            lr = build_lr_model([x[:3] for x in samples_with_t], [x[3] for x in samples_with_t])

            cnt = cnt + 1



def test():
    filename = 'data/kmean.dat'
    data = read_data(filename)
    array_data = np.array([x[:3] for x in data if x[0]])
    tri = build_triangulation(array_data)
    test = np.array([(5,1,1)])
    simplexes = find_simplex(tri, array_data, test)
    for i in range(len(simplexes)):
        prediction(data,simplexes[i],np.append(test[i],[1]))


def test_generate_data():
    filename = 'data/kmean.dat'
    resultfile = 'data/kmean_syn.dat'
    data = read_data(filename)
    array_data = np.array([x[:3] for x in data if x[0]])
    tri = build_triangulation(array_data)
    with open(resultfile, "w") as f:
        f.write("data memory vcore time" + "\n")  # quantity incategory
        for i in range(10,41):
            for j in range(20,51):
                for k in range(20,51):
                    test = np.array([i,j,k])
                    # print(test)
                    simplexes = find_simplex(tri, array_data, test)
                    result = prediction(data, simplexes, np.append(test, [1]))
                    # print(result)
                    f.write(str(result[0]) + ' ' + str(result[1]) + ' ' + str(result[2]) + ' ' + str(result[3]) + ' ' + "\n")


def test_lhs():
    feature_space = [[10,40],[20,50],[20,50]]
    sample_size = 1
    discrete_lhs(feature_space, sample_size)
    print(boundary(feature_space))

def test_driver():
    filename = 'data/kmean_syn.dat'
    ground_data = read_data(filename)
    feature_space = [[10, 40], [20, 50], [20, 50]]
    main_driver(feature_space, [[int(x[0]),int(x[1]),int(x[2]),float(x[3])] for x in ground_data])


# test()
test_driver()
# print(a-b)