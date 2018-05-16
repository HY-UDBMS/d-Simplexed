from pyDOE import *
import random

# number of decimal places to preserve in sample
sample_precision = 3

# scale from [old_min, old_max] --> [new_min, new_max]
def scale_to(old_min, old_max, new_min, new_max, value):
	return new_min + ((new_max - new_min)/(old_max - old_min))*(value - old_min)

def is_valid_sample(sample):
	# TODO: filter here to make sure samples leave some % overhead for the underlying system
	return True

# data = [[[a, b], runtime(a, b)], [[c, d], runtime(c, d)]
# sample = [e, f]
# find nearest [a, b] in data[i][0] by euclidean dist to sample and return it 
def find_nearest(data, sample):
	return None

# quicker way to LHS sample from a discrete set of features
# is easy to impl but has limitation that we can only return len(f1) samples
def seed_sample_v2():
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
		if not is_valid_sample(f1) or not is_valid_sample(f2):
			print "Invalid sample, please try again"
			sys.exit(-1)

		f1_sample = round(scale_to(0, 1, 0, 64, f1), sample_precision)
		f2_sample = round(scale_to(0, 1, 0, 64, f2), sample_precision)

		scaled_samples.append([f1_sample, f2_sample])

		print('%.3f' % f1_sample + "\t" + '%.3f' % f2_sample + "\t")

	return scaled_samples
