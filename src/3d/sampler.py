from pyDOE import *
from delaunaymodel import DelaunayModel
import random
import math

# number of decimal places to preserve in sample
sample_precision = 3

# scale from [old_min, old_max] --> [new_min, new_max]
def scale_to(old_min, old_max, new_min, new_max, value):
	return new_min + ((new_max - new_min)/(old_max - old_min))*(value - old_min)

# quicker way to LHS sample from a discrete set of features
# is easy to impl but has limitation that we can only return len(f1) samples
def seed_sample_v2(features):
	f1,f2 = features
	# sizeof f1 must equal sizeof f2
	f1 = [10, 20, 30, 40, 50]
	f2 = [5, 10, 15, 20, 25]

	random.shuffle(f1)
	random.shuffle(f2)

	lhs_samples = []
	while len(f1) > 0:
		lhs_samples.append([f1.pop(), f2.pop()])
	
	return lhs_samples

# seed_points = number of initial points to grab (must be < than |f1| and |f2|)
# f1 = [f1_min, f1_max]
# f2 = [f2_min, f2_max]
def seed_sample(seed_points, f1, f2):
	features = [f1, f2]

	# f2 = [0, 32] TODO get working as square before handling odd-sized spaces

	# the one with the biggest range is the one we need to normalize the others against for sampling

	print "Initial Latin Hypercube Samples"
	for i,f in enumerate(features):
		print("f" + str(i) + ": [{0}, {1}]".replace("{0}", str(f[0])).replace("{1}", str(f[1])))
	print "Seed point count: " + str(seed_points)
	print "----------------------"

	# refer: https://pythonhosted.org/pyDOE/randomized.html
	samples = lhs(len(features), seed_points, 'maximin')
	scaled_samples = []
	for f1,f2 in samples:
		f1_sample = round(scale_to(0, 1, 0, 64, f1), sample_precision)
		f2_sample = round(scale_to(0, 1, 0, 64, f2), sample_precision)

		scaled_samples.append([f1_sample, f2_sample])

		print('%.3f' % f1_sample + "\t" + '%.3f' % f2_sample + "\t")

	return scaled_samples

# model_points = points to create model with
# sample = [f1,f2],runtime -- sample to calculate utility for
def utility(model_points, sample):
	# construct DT model with model_points and sample
	# calculate utility using sample
	# utility in 3d = average (distance from current point w/ predicted runtime to the other 3 points that construct its prediction plane)
	print "constructing model for utility of sample " + str(sample)

	dt = DelaunayModel(model_points)
	dt.construct_model()

	est_runtime = dt.predict(sample[0])
	sample_hyperplane = dt.hyperplane_for(sample[0])

	print "util: find euclid dist between " + str(sample_hyperplane) + " --> " + str(sample)

	x_0, y_0, z_0 = sample[0][0], sample[0][1], sample[1]
	sum_dist = 0.0
	for x,y,z in sample_hyperplane:
		# euclidean dist between [x_0, y_0, z_0] and [x_1, y_1, z_1]: sqrt((x_1 - x_0)^2+(y_1 - y_0)^2+(z_1 - z_0)^2)
		sum_dist += (math.sqrt((x-x_0)**2 + (y-y_0)**2 + (z-z_0)**2))

	# utiliy = avg of distances to 3 points
	utility = sum_dist / 3.0
	print "util: util for " + str(sample) + " is " + str(utility)	

	return utility 

# current_model_points = points in current delaunay model
# available_points = points that can be added to the model
def next_adaptive_sample(current_model_points, available_points):
	print "next_adaptive_sample: sizeof current_model_points + available_points = " + str(len(current_model_points) + len(available_points))
	# after some threshold, you can't really LHS sample anymore because there are too many "holes" in the mesh
	# better to do it randomly after some threshold

	# re-lhs n samples
	# for each sample, predict runtime with same model
	# then take dist between point and 3-nearest neighbors in 3d space
	# the point with the highest distance wins

	# but for now, just do rand
	#random.shuffle(available_points)

	utils = {}
	for i,hist_point in enumerate(available_points):
		cfg = hist_point[0]
		runtime = hist_point[1]
		print str(i) + " checking " + str(cfg) + " --> " +str(runtime)
		utils[i] = utility(current_model_points, hist_point)

	print "determined utils: " + str(utils)
	highest_util_idx = 0
	highest_util_val = utils[0]
	for key,val in utils.items():
		if val > highest_util_val:
			highest_util_val = val
			highest_util_idx = key

	print "LARGEST utility ({}) belongs to {}".format(highest_util_val, available_points[highest_util_idx])
	return available_points.pop(highest_util_idx)

