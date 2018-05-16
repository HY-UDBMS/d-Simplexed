from delaunaymodel import DelaunayModel
import sampler


# vmem, vcore --> runtime
raw_hist_data = """
10	5	6.3687
20	5	4.33018
30	5	1.49257
40	5	1.38429
50	5	1.37552
10	10	5.18841
20	10	2.87083
30	10	1.08855
40	10	1.10772
50	10	1.01886
10	15	5.15312
20	15	2.66598
30	15	0.93767
40	15	0.90635
50	15	1.06719
10	20	3.26359
20	20	2.2035
30	20	0.85163
40	20	0.93339
50	20	0.91893
10	25	3.53785
20	25	2.13402
30	25	0.94102
40	25	0.89912
50	25	0.88065"""

hist_data = []
for line in raw_hist_data.split('\n'):
	if not line:
		continue

	split = line.split()
	hist_data.append([[int(split[0]), int(split[1])], float(split[2])])

def lookup_runtime(rt_cfg, hist_data):
	for hist_cfg,runtime in hist_data:
		if rt_cfg == hist_cfg:
			return runtime
	return None

print "Size of historical data pool: " + str(len(hist_data))
print hist_data

seed_samples = sampler.seed_sample_v2();
seed_samples = [[[f1,f2], lookup_runtime([f1,f2], hist_data)] for [f1, f2] in seed_samples]
seed_count = len(seed_samples)

print "Number of seed samples from sampler: " + str(seed_count)
print "Seed samples from sampler: " + str(seed_samples)

print "Adding in boundary samples..."
seed_samples.append([[10,5], lookup_runtime([10,5], hist_data)]);
seed_samples.append([[10,25], lookup_runtime([10,25], hist_data)]);
seed_samples.append([[50,5], lookup_runtime([50,5], hist_data)]);
seed_samples.append([[50,25], lookup_runtime([50,25], hist_data)]);

# TODO: remove seed_samples from hist_data
for seed in seed_samples:
	for i,data in enumerate(hist_data):
		if data[0][0] == seed[0][0] and data[0][1] == seed[0][1]:
			print "Removing from historical data pool: " + str(data)
			del hist_data[i]

print "Size of historical data pool: " + str(len(hist_data))

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
while len(hist_data) > 1:
	# grab next sample and remove from hist_data
	print "len(hist_data) BEFORE " + str(len(hist_data))
	next_sample = sampler.next_adaptive_sample(hist_data)
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



