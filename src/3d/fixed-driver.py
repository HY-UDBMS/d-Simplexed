from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn import linear_model
import numpy as np
from delaunaymodel import DelaunayModel
import sampler
import random
import sys

"""
This script saves *test_set_ratio* of the historical runtime data to use as a testing set
Sorry, it is a little unorganized.

It reads an input file containing historical test data, and from that, (1) takes the initial seed samples, then (2) builds the DT model, (3) compare MAPE using Gaussian Process method, (4) compare MAPE using linear regression method, (5) compare MAPE using Delaunay Triangulation method.

After the initial sampels have been selected, we build the 3 models and then add points to them in *iterative_step_size* steps, then report the MAPE accuracy for each.

You can run and pipe the results into a CSV file easily by doing (e.g.):

python fixed-driver.py wc80g.csv | grep --line-buffered -E "\[Reporting"
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

args = sys.argv[1:]

if not len(args) == 1:
	print "Usage: fixed-driver.py <input-file>"
	sys.exit(-1)

# f1<tab>f2<tab>runtime(f1,f2)
hist_data_file = args[0]

# how many points to add to the model with each iteration
iterative_step_size = 5

# percent (as decimal) of hist_data to reserve for testing
test_set_ratio = 0.10

print "Reading input from file: " + hist_data_file

# read from file --> hist_data
hist_data = []
with open(hist_data_file, "r") as ins:
	for line in ins:
		split = line.split()
		hist_data.append([[int(split[0]), int(split[1])], round(100*float(split[2]), 1)])

input_len = len(hist_data)
test_set_count = int(input_len * test_set_ratio)
print "Size of historical data pool: " + str(input_len)

# TODO: make this dynamic
# len(f1) must equal len(f2)
#f1_range = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
#f2_range = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
f1_range = np.arange(40,242,2).tolist()
f2_range = np.arange(60,161,1).tolist()
#f1_range = np.arange(1,121,1).tolist()
#f2_range = np.arange(1,121,1).tolist()
#f1_range = np.arange(1,101,1).tolist()
#f2_range = np.arange(1,101,1).tolist()
feature_space = [f1_range,f2_range]

# < 4 (8 total points) and you run into problems forming the triangulation
#seed_lhs_count = 4 
#seed_samples = sampler.seed_sample([f1_range, f2_range], seed_lhs_count);

seed_samples = []
#grid_samples = sampler.get_gridding_samples(feature_space, [40,20,10,5,4,2,1])

#print "grid_samples=" + str(grid_samples)

# grab 8 grid samples
random.shuffle(hist_data)
for i in range(4):
	seed_samples.append(hist_data.pop())
#	seed_samples.append(grid_samples.pop(0))

#seed_samples = [[[f1,f2], lookup_runtime([f1,f2], hist_data)] for [f1, f2] in seed_samples]
seed_count = len(seed_samples)

print "Number of seed samples from sampler: " + str(seed_count)
print "Seed samples from sampler: " + str(seed_samples)
print "Adding in boundary samples..."
boundary_samples = [[f1_range[0],f2_range[0]],[f1_range[0],f2_range[-1]],[f1_range[-1],f2_range[0]],[f1_range[-1],f2_range[-1]]]
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
#	next_for_test_set = hist_data[0]
	#if the point we want to save for the test set is in seed_samples, don't add it
#	if next_for_test_set not in seed_samples:
	test_set.append(hist_data.pop(0))

print "len(test_set) = " + str(len(test_set))
print "len(hist_data) = " + str(len(hist_data))
print "Size of historical data pool: " + str(len(hist_data))
print "Size of seed_samples: " + str(len(seed_samples))
print "hist_data before subtracting seed_samples" + str(hist_data)
print "seed_samples here: " + str(seed_samples)
print "len(hist_data) " + str(len(hist_data))
print "len(test_set) " + str(len(test_set))
print "len(seed_samples) " + str(len(seed_samples))
print "hist_data + test_set + seed_samples = " + str(len(hist_data) + len(test_set) + len(seed_samples))

assert (len(hist_data) + len(test_set) + len(seed_samples)) == input_len

#######################################
# Build the model with the seed points
#######################################


# Instanciate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-5, 1e5))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)

gp_training_X = [[f1,f2] for [[f1,f2],runtime] in seed_samples]
gp_training_Y = [[runtime] for [[f1,f2],runtime] in seed_samples]

gp_testset_X = [[f1,f2] for [[f1,f2],runtime] in test_set]
gp_testset_Y = [[runtime] for [[f1,f2],runtime] in test_set]

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(gp_training_X, gp_training_Y)
y_pred, sigma = gp.predict(gp_testset_X, return_std=True)

total_err = 0 
sample_count = 0
for idx,runtime_val in enumerate(gp_testset_Y):
	est_runtime = y_pred[idx]
	actual_runtime = runtime_val
	pct_err = round(abs((actual_runtime - est_runtime) / actual_runtime), 5)
	total_err += pct_err
	sample_count += 1
	#print "[GP]{}: Predicted={} Actual={}, MAPE={}%".format(gp_testset_X[idx], est_runtime, actual_runtime, pct_err * 100)

mape_gp = round(total_err/sample_count*100,2)

if False:
	# Use basic multivariate linear regression (X,Y)-->Z
	lm = linear_model.LinearRegression()
	model = lm.fit(gp_training_X, gp_training_Y)

	# Make the prediction on the meshed x-axis (ask for MSE as well)
	y_pred = lm.predict(gp_testset_X)
	total_err = 0 
	sample_count = 0
	for idx,runtime_val in enumerate(gp_testset_Y):
		est_runtime = y_pred[idx]
		actual_runtime = runtime_val
		pct_err = round(abs((actual_runtime - est_runtime) / actual_runtime), 5)
		total_err += pct_err
		sample_count += 1

mape_lr = 0
#mape_lr = round(total_err/sample_count*100,2)

# build initial model with seed_samples
dt = DelaunayModel(seed_samples)
dt.construct_model()

# determine overall MAPE of model with just seed samples
total_err = 0 
sample_count = 0
for [a,b],c in test_set:
	print "predicting for " + str([a, b])
	est_runtime = dt.predict([a,b])
	actual_runtime = lookup_runtime([a,b], test_set);
	pct_err = round(abs((actual_runtime - est_runtime) / actual_runtime), 5)
	total_err += pct_err
	sample_count += 1
	print "{}: Predicted={} Actual={}, MAPE={}%".format([a,b], est_runtime, actual_runtime, pct_err * 100)

mape_dt = round(total_err/sample_count*100,2)

print "[Reporting] {}\t{}\t{}\t{}".format(len(seed_samples), mape_dt, mape_gp, mape_lr)

# then add 1-by-1 new points via adaptive sampling, re-building the model and noting the error
#######################################
# Iteratively add points to model and test
#######################################

model_samples = list(seed_samples)
# stop at len_hist(data) == 3 because we can't build a model with < 3 points
while len(hist_data) > 3:
	# grab next sample and remove from hist_data
	# add 5 at a time
	for i in range(iterative_step_size):
		#found_sample = False
		#while not found_sample:
		#	next_sample = grid_samples.pop(0)
		#	if next_sample not in [x[0] for x in test_set]:
				# if next_sample in test set, ignore it
		#		found_sample = True

		#next_sample = sampler.next_adaptive_sample(model_samples, hist_data, feature_space)
		#next_sample = [next_sample,lookup_runtime(next_sample, hist_data)]
		next_sample = hist_data.pop() 
		model_samples.append(next_sample)
		
	print "appending..." + str(next_sample)

	dt = DelaunayModel(model_samples)
	dt.construct_model()

	# Instanciate a Gaussian Process model
	kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-5, 1e5))
	#gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)

	gp_training_X = [[f1,f2] for [[f1,f2],runtime] in model_samples]
	gp_training_Y = [[runtime] for [[f1,f2],runtime] in model_samples]

	gp_testset_X = [[f1,f2] for [[f1,f2],runtime] in test_set]
	gp_testset_Y = [[runtime] for [[f1,f2],runtime] in test_set]

	# Fit to data using Maximum Likelihood Estimation of the parameters
	#gp.fit(gp_training_X, gp_training_Y)

	#y_pred, sigma = gp.predict(gp_testset_X, return_std=True)

	total_err = 0 
	sample_count = 0
	for idx,runtime_val in enumerate(gp_testset_Y):
		est_runtime = y_pred[idx]
		actual_runtime = runtime_val
		pct_err = round(abs((actual_runtime - est_runtime) / actual_runtime), 5)
		total_err += pct_err
		sample_count += 1

	mape_gp = round(total_err/sample_count*100,2)
	#mape_gp = 0
	# Use basic multivariate linear regression (X,Y)-->Z
	if False:
		lm = linear_model.LinearRegression()
		model = lm.fit(gp_training_X, gp_training_Y)

		# # Make the prediction on the meshed x-axis (ask for MSE as well)
		y_pred = lm.predict(gp_testset_X)
		total_err = 0 
		sample_count = 0
		for idx,runtime_val in enumerate(gp_testset_Y):
			est_runtime = y_pred[idx]
			actual_runtime = runtime_val
			pct_err = round(abs((actual_runtime - est_runtime) / actual_runtime), 5)
			total_err += pct_err
			sample_count += 1

	mape_lr = 0
	#mape_lr = round(total_err/sample_count*100, 2)

	total_err = 0 
	sample_count = 0
	for [a,b],c in test_set:
		est_runtime = dt.predict([a,b])
		actual_runtime = lookup_runtime([a,b], test_set);
		pct_err = round(abs((actual_runtime - est_runtime) / actual_runtime), 5)
		total_err += pct_err
		sample_count += 1
		print "{}: Predicted={} Actual={}, MAPE={}%".format([a,b], est_runtime, actual_runtime, pct_err * 100)

	mape_dt = round(total_err/sample_count*100,2)
	
	print "[Reporting] {}\t{}\t{}\t{}".format(len(model_samples), mape_dt, mape_gp, mape_lr)
