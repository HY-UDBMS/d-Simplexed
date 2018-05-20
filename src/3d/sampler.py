from pyDOE import *
from delaunaymodel import DelaunayModel
import random
import math

# number of decimal places to preserve in sample
sample_precision = 3

# scale from [old_min, old_max] --> [new_min, new_max]
def scale_to(old_min, old_max, new_min, new_max, value):
	return new_min + ((new_max - new_min)/(old_max - old_min))*(value - old_min)

# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

# quicker way to LHS sample from a discrete set of features
# is easy to impl but has limitation that we can only return len(f1) samples
# TODO: optimize to handle when sizeof f1 != sizeof f2 (scaling needed)
def seed_sample(features, samples):
	f1,f2 = features

	# sizeof f1 must equal sizeof f2
	assert len(f1) == len(f2)

	lhs_samples = discrete_lhs([], features, samples)
	return lhs_samples

# taken_points = points in points_space already used in model as array of [[f1,f2]runtime]
# feature_space = f1=[a, b, c, ..., n], f2=[a, b, c, d, ..., n] -- total range of features
# samples = number of samples to take
def discrete_lhs(taken_points, feature_space, samples):
	f1,f2 = feature_space

	# sizeof f1 must equal sizeof f2
	assert len(f1) == len(f2)

	print "discrete_lhs samples to take: " + str(samples)

	# divide feature_space into buckets
	f1_buckets = chunkIt(f1, samples)
	f2_buckets = chunkIt(f2, samples)
	
	print "f1_buckets " + str(f1_buckets)
	print "f2_buckets " + str(f2_buckets)

	random.shuffle(f1_buckets)
	random.shuffle(f2_buckets)

	lhs_sample_buckets = []
	while len(f1_buckets) > 0:
		lhs_sample_buckets.append([f1_buckets.pop(), f2_buckets.pop()])

	# each lhs_sample_buckets[i] is something like [[a,b,c][d,e,f]], which represents a square-ish range of the domain that we need to pick a sample from
	# generate the possible points, then return the first one that is not in taken_points
	print str(lhs_sample_buckets)

	lhs_samples = []
	for bucket_pair in lhs_sample_buckets:
		f1_buckets_i,f2_buckets_i = bucket_pair
		print "f1_buckets_i" + str(f1_buckets_i)
		print "f2_buckets_i" + str(f2_buckets_i)

		candidates = []
		for f1_vals in f1_buckets_i:
			for f2_vals in f2_buckets_i:
				candidates.append([f1_vals, f2_vals])
		
		for candidate in candidates:
			f1,f2 = candidate
			# check if point is already taken (in model)
			def is_taken(points, point):
				for taken_point in points:
					if taken_point[0] == point:
						print "point " + str(point) + " is taken!"
						return True
				return False
	
			if not is_taken(taken_points, candidate):
				lhs_samples.append(candidate)
				break

	print "discrete_lhs final samples: " + str(lhs_samples)

	return lhs_samples

# model_points = points to create model with
# sample = [f1,f2] -- sample to calculate utility for
def utility(model_points, sample):
	# construct DT model with model_points
	# calculate utility using sample
	# utility in 3d = average (distance from current point w/ predicted runtime to the other 3 points that construct its prediction plane)
	print "constructing model for utility of sample " + str(sample)

	dt = DelaunayModel(model_points)
	dt.construct_model()

	est_runtime = dt.predict(sample)
	sample_hyperplane = dt.hyperplane_for(sample)

	print "util: find euclid dist between " + str(sample_hyperplane) + " --> " + str(sample)

	x_0, y_0, z_0 = sample[0], sample[1], est_runtime
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
# feature_space = possible f1 and f2 values, eg [[1,2,3...n][5,10,15,...n]]
def next_adaptive_sample(current_model_points, available_points, feature_space):
	print "next_adaptive_sample: sizeof current_model_points + available_points = " + str(len(current_model_points) + len(available_points))
	# after some threshold, you can't really LHS sample anymore because there are too many "holes" in the mesh
	# better to do it randomly after some threshold

	# re-lhs n samples
	# for each sample, predict runtime with same model

	next_samples = []

	# for md model, the LHS samples we get from lhs_sample_v2 is 21
	lhs_sample_size = 24
	if len(available_points) < lhs_sample_size:
		# not enough points to get a full LHS sample, so we'll consider the rest at once (edge case)
		print "Not enough points remain to LHS"
		next_samples = [[f1,f2] for [[f1,f2],rt] in available_points]
	else:
		#random.shuffle(available_points)
		#next_samples = available_points[:10] # grab 10 at random
		print "discrete_lhs current_model_points" + str(current_model_points)
		print "discrete_lhs feature_space" + str(feature_space)
		next_samples = discrete_lhs(current_model_points, feature_space, lhs_sample_size)
		# discrete_lhs returns [f1,f2] values, need to attach actual runtime

	utils = {}
	for i,hist_point in enumerate(next_samples):
		cfg = hist_point
		#runtime = hist_point[1]
		#print str(i) + " checking " + str(cfg) + " --> " +str(runtime)
		utils[i] = utility(current_model_points, cfg)

	# if utils are empty, it means we couldn't get enough samples for a full discrete_lhs, so in that case just return the rest randomized (it should be very few points relatively anyway)
	if len(utils) == 0:
		random.shuffle(available_points)
		return available_points.pop()

	print "determined utils: " + str(utils)
	# sort highest util
	highest_util_idx = 0
	highest_util_val = utils[0]
	for key,val in utils.items():
		if val > highest_util_val:
			highest_util_val = val
			highest_util_idx = key

	print "LARGEST utility ({}) belongs to {}".format(highest_util_val, available_points[highest_util_idx])
	popped = available_points.pop(highest_util_idx)
	return popped
