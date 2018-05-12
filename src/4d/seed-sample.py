from pyDOE import *

# number of decimal places to preserve in sample
sample_precision = 3

# scale from [old_min, old_max] --> [new_min, new_max]
def scale_to(old_min, old_max, new_min, new_max, value):
	return new_min + ((new_max - new_min)/(old_max - old_min))*(value - old_min)

def is_valid_sample(sample):
	# TODO: filter here to make sure samples leave some % overhead for the underlying system
	return True

f1 = [0, 64]
f2 = [0, 64]
features = [f1, f2]

# number of initial points to grab (must be < than |f1| and |f2|)
seed_points = 15;

# f2 = [0, 32] TODO get working as square before handling odd-sized spaces

# the one with the biggest range is the one we need to normalize the others against for sampling

print "Initial Latin Hypercube Samples"
print "f1: [{0}, {1}]".replace("{0}", str(f1[0])).replace("{1}", str(f1[1]))
print "f2: [{0}, {1}]".replace("{0}", str(f2[0])).replace("{1}", str(f2[1]))
print "Seed point count: " + str(seed_points)
print "----------------------"

# refer: https://pythonhosted.org/pyDOE/randomized.html
samples = lhs(len(features), seed_points, 'maximin')
for f1,f2 in samples:
	if not is_valid_sample(f1) or not is_valid_sample(f2):
		print "Invalid sample, please try again"
		sys.exit(-1)

	f1_sample = scale_to(0, 1, 0, 64, f1)
	f2_sample = scale_to(0, 1, 0, 64, f2)

	print('%.3f' % f1_sample + "\t" + '%.3f' % f2_sample)


