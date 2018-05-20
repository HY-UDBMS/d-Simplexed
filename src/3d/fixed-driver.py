from delaunaymodel import DelaunayModel
import sampler
import random

"""
This script saves 20% of the historical runtime data to use as a testing set
"""

def lookup_runtime(rt_cfg, hist_data):
	for hist_cfg,runtime in hist_data:
		if rt_cfg == hist_cfg:
			return runtime
	return None

def contains_sample(samples, sample):
	for a,runtime in samples:
		if a == sample:
			return True
	return False

# f1<tab>f2<tab>runtime(f1,f2)
#hist_data_file = "hist-5g-sm.csv"
hist_data_file = "hist-5g-md.csv"

print "Reading input from file: " + hist_data_file

# read from file --> hist_data
hist_data = []
with open(hist_data_file, "r") as ins:
	for line in ins:
		split = line.split()
		hist_data.append([[int(split[0]), int(split[1])], round(100*float(split[2]), 1)])

input_len = len(hist_data)
test_set_count = int(input_len * 0.2)
print "Size of historical data pool: " + str(input_len)

# TODO: make this dynamic
# len(f1) must equal len(f2)
f1 = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
f2 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
feature_space = [f1,f2]
#f1 = [10, 20, 30, 40, 50]
#f2 = [5, 10, 15, 20, 25]


# < 4 (8 total points) and you run into problems forming the triangulation
seed_lhs_count = 4 
seed_samples = sampler.seed_sample([f1, f2], seed_lhs_count);
seed_samples = [[[f1,f2], lookup_runtime([f1,f2], hist_data)] for [f1, f2] in seed_samples]
seed_count = len(seed_samples)

print "Number of seed samples from sampler: " + str(seed_count)
print "Seed samples from sampler: " + str(seed_samples)
print "Adding in boundary samples..."
boundary_samples = [[10,5],[10,25],[50,5],[50,25]]
for boundary_sample in boundary_samples:
	if not contains_sample(seed_samples, boundary_sample):
		print "add boundary " + str(boundary_sample)
		seed_samples.append([boundary_sample, lookup_runtime(boundary_sample, hist_data)]);

print "finished seed samples: " + str(seed_samples)
print "len(seed_samples) after everything: " + str(len(seed_samples))

# subtract seed_samples from hist_data
for seed in seed_samples:
	print "Searching in hist_data for seed=" + str(seed)
	found = False
	for i,data in enumerate(hist_data):
		if data[0][0] == seed[0][0] and data[0][1] == seed[0][1]:
			print "Removing from historical data pool: " + str(data)
			del hist_data[i]
			found = True
	if not found:
		print "seed NOT found!!!"

# grab 20% of points at random to preserve for only testing, these do not go into model
random.shuffle(hist_data)
test_set = []
while len(test_set) < test_set_count:
	next_tester = hist_data[-1]
	# if we happen to pick at random a seed point, exclude it
	#already_in_seeds = False
	#for seed_sample in seed_samples:
#		if seed_sample[0] == next_tester:
#			already_in_seeds = True
	
#	if not already_in_seeds:
	test_set.append(hist_data.pop())

print "len(test_set) = " + str(len(test_set))
print "len(hist_data) = " + str(len(hist_data))
print "Size of historical data pool: " + str(len(hist_data))
print "Size of seed_samples: " + str(len(seed_samples))
print "hist_data before subtracting seed_samples" + str(hist_data)
print "seed_samples here: " + str(seed_samples)


# TODO: make sure these line-up, test_set should be exactly 20% of hist_data + seed_samples
print "len(hist_data) " + str(len(hist_data))
print "len(test_set) " + str(len(test_set))
print "len(seed_samples) " + str(len(seed_samples))
print "hist_data + test_set + seed_samples = " + str(len(hist_data) + len(test_set) + len(seed_samples))

assert (len(hist_data) + len(test_set) + len(seed_samples)) == input_len

# build initial model with seed_samples
dt = DelaunayModel(seed_samples)
dt.construct_model()

# determine overall MPE of model with just seed samples
total_err = 0 
sample_count = 0
for [a,b],c in test_set:
	print "predicting for " + str([a, b])
	est_runtime = dt.predict([a,b])
	actual_runtime = lookup_runtime([a,b], test_set);
	pct_err = round((actual_runtime - est_runtime) / actual_runtime, 5)
	total_err += pct_err
	sample_count += 1
	print "{}: Predicted={} Actual={}, MPE={}%".format([a,b], est_runtime, actual_runtime, pct_err * 100)

#############
#print "Mean percent error over {} remaining samples: {}%".format(sample_count, round(total_err/sample_count * 100, 2))
print "[Reporting] {}\t{}".format(sample_count, round(total_err/sample_count*100,2))
#############

# then add 1-by-1 new points via adaptive sampling, re-building the model and noting the error

model_samples = list(seed_samples)
# stop at len_hist(data) == 3 because we can't build a model with < 3 points
while len(hist_data) > 3:
	# grab next sample and remove from hist_data
	print "len(hist_data) BEFORE " + str(len(hist_data))
	feature_space = [[10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50],[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]]
	next_sample = sampler.next_adaptive_sample(model_samples, hist_data, feature_space)
	print "len(hist_data) AFTER " + str(len(hist_data))

	print "next_sample=" + str(next_sample)
	model_samples.append(next_sample)

	dt = DelaunayModel(model_samples)
	dt.construct_model()

	total_err = 0 
	sample_count = 0
	for [a,b],c in test_set:
		est_runtime = dt.predict([a,b])
		actual_runtime = lookup_runtime([a,b], test_set);
		pct_err = round((actual_runtime - est_runtime) / actual_runtime, 5)
		total_err += pct_err
		sample_count += 1
		print "{}: Predicted={} Actual={}, MPE={}%".format([a,b], est_runtime, actual_runtime, pct_err * 100)

	#############
	#print "Mean percent error over {} remaining samples: {}%".format(sample_count, round(total_err/sample_count * 100, 2))
	print "[Reporting] {}\t{}\t{}".format(len(model_samples), sample_count, round(total_err/sample_count*100,2))
	#############
