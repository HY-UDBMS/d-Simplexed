from delaunaymodel import DelaunayModel
import numpy as np
import sampler
import random

"""
This is the same as 3d-driver, but we avoid using LHS during the model build
This is only for generating more 80g data
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
hist_data_file = "hist-80g-sm.csv"

# generate data at the end
do_datagen = True

print "Reading input from file: " + hist_data_file

# read from file --> hist_data
hist_data = []
with open(hist_data_file, "r") as ins:
	for line in ins:
		split = line.split()
		hist_data.append([[int(split[0]), int(split[1])], round(100*float(split[2]), 1)])

print "hist_data=" + str(hist_data)
random.shuffle(hist_data)
dt = DelaunayModel(hist_data)
dt.construct_model()

if do_datagen:
	# with constructed model, use it to predict more theoretical runtimes
	f1_min = 40
	f1_max = 240
	f1_stepsize = 2

	f2_min = 60
	f2_max = 160
	f2_stepsize = 1

	for f1_i in range(f1_min, f1_max+1, f1_stepsize):
		for f2_i in range(f2_min, f2_max+f2_stepsize):
			# use this to grab output values
			# python 3d-driver.py | grep datagen | cut -c 9- > out.dat
			print "datagen:{}\t{}\t{}".format(f1_i, f2_i, round(dt.predict([f1_i, f2_i]),2))
