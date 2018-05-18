import sampler

f1 = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
f2 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

taken_points = [[14,12],[10,7]]
features = [f1, f2]
samples = 8

print "LHS samples " + str(sampler.discrete_lhs(taken_points, features, samples))
