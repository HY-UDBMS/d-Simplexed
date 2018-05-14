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
f3 = [0, 64]
features = [f1, f2, f3]

# number of initial points to grab (must be < than |f1| and |f2|)
seed_points = 15;

# f2 = [0, 32] TODO get working as square before handling odd-sized spaces

# the one with the biggest range is the one we need to normalize the others against for sampling

print "Picking Adaptive Latin Hypercube Sample"
for i,f in enumerate(features):
	print("f" + str(i) + ": [{0}, {1}]".replace("{0}", str(f[0])).replace("{1}", str(f[1])))
print "Seed point count: " + str(seed_points)
print "----------------------"

# refer: https://pythonhosted.org/pyDOE/randomized.html
samples = lhs(len(features), seed_points, 'maximin')
for f1,f2,f3 in samples:
	if not is_valid_sample(f1) or not is_valid_sample(f2) or not is_valid_sample(f3):
		print "Invalid sample, please try again"
		sys.exit(-1)

	f1_sample = scale_to(0, 1, 0, 64, f1)
	f2_sample = scale_to(0, 1, 0, 64, f2)
	f3_sample = scale_to(0, 1, 0, 64, f3)

	print('%.3f' % f1_sample + "\t" + '%.3f' % f2_sample + "\t" + '%.3f' % f3_sample)


