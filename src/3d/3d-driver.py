from delaunaymodel import DelaunayModel
import sampler


# vmem, vcore --> runtime
hist_data_file = "hist-5g-sm.csv"

hist_data = []
with open(hist_data_file, "r") as ins:
	for line in ins:
		split = line.split()
		hist_data.append([[int(split[0]), int(split[1])], round(100*float(split[2]), 1)])

def lookup_runtime(rt_cfg, hist_data):
	for hist_cfg,runtime in hist_data:
		if rt_cfg == hist_cfg:
			return runtime
	return None

print "Size of historical data pool: " + str(len(hist_data))
print hist_data


f1 = [10, 20, 30, 40, 50]
f2 = [5, 10, 15, 20, 25]

seed_samples = sampler.seed_sample_v2([f1, f2]);
seed_samples = [[[f1,f2], lookup_runtime([f1,f2], hist_data)] for [f1, f2] in seed_samples]
seed_count = len(seed_samples)

def contains_sample(samples, sample):
	for a,runtime in samples:
		if a == sample:
			return True
	return False

print "Number of seed samples from sampler: " + str(seed_count)
print "Seed samples from sampler: " + str(seed_samples)

print "Adding in boundary samples..."
if not contains_sample(seed_samples, [10, 5]):
	seed_samples.append([[10,5], lookup_runtime([10,5], hist_data)]);

if not contains_sample(seed_samples, [10, 25]):
	seed_samples.append([[10,25], lookup_runtime([10,25], hist_data)]);

if not contains_sample(seed_samples, [50, 5]):
	seed_samples.append([[50,5], lookup_runtime([50,5], hist_data)]);

if not contains_sample(seed_samples, [50, 25]):
	seed_samples.append([[50,25], lookup_runtime([50,25], hist_data)]);

print "Size of historical data pool: " + str(len(hist_data))
print "Size of seed_samples: " + str(len(seed_samples))

print "hist_data before subtracting seed_samples" + str(hist_data)

print "seed_samples here: " + str(seed_samples)

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
	

assert len(hist_data) + len(seed_samples) == 25

print "Size of historical data pool: " + str(len(hist_data))
print "Size of seed_samples: " + str(len(seed_samples))

# build initial model with seed_samples
dt = DelaunayModel(seed_samples)
dt.construct_model()

total_err = 0 
sample_count = 0
for [a,b],c in hist_data:
	est_runtime = dt.predict([a,b])
	actual_runtime = lookup_runtime([a,b], hist_data);
	pct_err = round((actual_runtime - est_runtime) / actual_runtime, 5)
	total_err += pct_err
	sample_count += 1
	print "{}: Predicted={} Actual={}, MPE={}%".format([a,b], est_runtime, actual_runtime, pct_err * 100)

#############
print "Mean percent error over {} remaining samples: {}%".format(sample_count, round(total_err/sample_count * 100, 2))
#############

# then add 1-by-1 new points via adaptive sampling, re-building the model and noting the error

model_samples = list(seed_samples)
# stop at len_hist(data) == 3 because we can't build a model with < 3 points
while len(hist_data) > 3:
	# grab next sample and remove from hist_data
	print "len(hist_data) BEFORE " + str(len(hist_data))
	next_sample = sampler.next_adaptive_sample(model_samples, hist_data)
	print "len(list_data) AFTER " + str(len(hist_data))

	print "next_sample=" + str(next_sample)
	model_samples.append(next_sample)

	dt = DelaunayModel(model_samples)
	dt.construct_model()

	total_err = 0 
	sample_count = 0
	for [a,b],c in hist_data:
		est_runtime = dt.predict([a,b])
		actual_runtime = lookup_runtime([a,b], hist_data);
		pct_err = round((actual_runtime - est_runtime) / actual_runtime, 5)
		total_err += pct_err
		sample_count += 1
		print "{}: Predicted={} Actual={}, MPE={}%".format([a,b], est_runtime, actual_runtime, pct_err * 100)

	#############
	print "Mean percent error over {} remaining samples: {}%".format(sample_count, round(total_err/sample_count * 100, 2))
	#############
# get next point via adaptive sampling
#while len(hist_data) > 0:
	# pop next and go

# build COMPLETE model with all hist data
model_samples = model_samples + hist_data
print "building final model with samples " + str(model_samples)
print "len(model_samples)=" + str(len(model_samples))
dt = DelaunayModel(model_samples)
dt.construct_model()

# with constructed model, use it to predict more theoretical runtimes
f1_min = 10
f1_max = 50

f2_min = 5
f2_max = 25

for f1_i in range(f1_min, f1_max+1):
	for f2_i in range(f2_min, f2_max+1):
		# use this to grab output values
		# python 3d-driver.py | grep datagen | cut -c 9- > out.dat
		print "datagen:{}\t{}\t{}".format(f1_i, f2_i, dt.predict([f1_i, f2_i]))
