"""
    A utility script for randomly sampling n samples from a vmmem x vcore space.
"""
import sys
import random

def is_divisible_by(nums, n):
	for num in nums:
		if n % num == 0:
			return True

# pick n random pairs of (vmem, vcores) execution configurations within the bounds of xmin, xmax, ymin and ymax
def get_rand_points(xmin, xmax, ymin, ymax, n, out_format):
	rand_points = []

	for i in range(n):
		divides = False

		# Ensure vcores (y) are divisible by at least 2, 3, 4 or 5, so that we have that many containers, not more or less
		while not divides:
			vcores = random.randint(ymin, ymax)
			divides = is_divisible_by([5, 4, 3, 2], vcores)

		vmem = random.randint(xmin, xmax)
		rand_points.append((vmem, vcores))

	for vmem, vcores in rand_points:
		for divisor in [5, 4, 3, 2]:
			if vcores % divisor == 0:
				# found divider
				# vmem (x) can be divided into mb, but vcores cannot be split
				executors = divisor
				vcores_per_exec = vcores / divisor
				vmem_per_exec = vmem * 1000 / divisor 

				if out_format == "pretty":
					print(str(vcores) + "v" + str(vmem) + "g --> " + str(executors) + " execs, " + str(vcores_per_exec) + " cores/exec, " + str(vmem_per_exec) + "mb vmem/exec");
				elif out_format == "hibench":
					print(str(executors) + "\t" + str(vcores_per_exec) + "\t" + str(vmem_per_exec) + "mb");
				else:
					print "invalid output format: " + out_format
				break

if len(sys.argv) < 3:
	print "Usage: rand-sample.py sample-num output=hibench|output=pretty" 
	sys.exit()

n = int(sys.argv[1])
out_format = sys.argv[2].replace("output=", "")

# select from 2-56gb vmem and 2-15 vcores (need to remember to leave some resources for the host)
get_rand_points(2, 56, 2, 15, n, out_format)
