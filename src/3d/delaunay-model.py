from pyhull.delaunay import DelaunayTri
import numpy as np

# f1,f2,runtime(f1, f2)
model_points_full = [[0,0,100],
				[0,32,40],
				[32,0,40],
				[32,32,40],
				[8,8,65],
				[16,24,55]]

unknowns = [[16,8],
			[4,24],
			[24,4],
			[4,4]]

def get_runtime(point):
	for f1,f2,runtime in model_points_full:
		if point[0] == f1 and point[1] == f2:
			return runtime

# refer: http://kitchingroup.cheme.cmu.edu/blog/2015/01/18/Equation-of-a-plane-through-three-points/
def calc_hyperplane(p1, p2, p3):
	p1 = np.array(p1)
	p2 = np.array(p2)
	p3 = np.array(p3)

	# These two vectors are in the plane
	v1 = p3 - p1
	v2 = p2 - p1

	# the cross product is a vector normal to the plane
	cp = np.cross(v1, v2)
	a, b, c = cp

	# This evaluates a * x3 + b * y3 + c * z3 which equals d
	d = np.dot(cp, p3)

	print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
	print('The equation is ({3} - {0}x - {1}y)/{2}'.format(a, b, c, d))

	return [a, b, c, d]

# use hyperplane [a, b, c, d] to make prediction for point [f1, f2]
def make_prediction(hyperplane, point):
	a,b,c,d = hyperplane
	return (d - a*point[0] - b*point[1])/c


# build model w/o runtime
model_points = [[f1,f2] for [f1,f2,f3] in model_points_full]

# do triangulation with model_points
# refer: https://pythonhosted.org/pyhull/
tri = DelaunayTri(model_points)

# print("tri " + str(tri.simplices))

for simplex in tri.simplices:
	p1 = [simplex.coords[0].item(0), simplex.coords[0].item(1), get_runtime([simplex.coords[0].item(0), simplex.coords[0].item(1)])]
	p2 = [simplex.coords[1].item(0), simplex.coords[1].item(1), get_runtime([simplex.coords[1].item(0), simplex.coords[1].item(1)])]
	p3 = [simplex.coords[2].item(0), simplex.coords[2].item(1), get_runtime([simplex.coords[2].item(0), simplex.coords[2].item(1)])]

	for f1,f2 in unknowns:
		if simplex.in_simplex([f1, f2]):
			print "need to predict for " + str([f1, f2])
			print("TODO calculate hyperplane for <{}, {}, {}> <{}, {}, {}> <{}, {}, {}>".format(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2]))

			hyperplane = calc_hyperplane(p1, p2, p3)
			predicted_runtime = make_prediction(hyperplane, [f1, f2])
			print "predicted runtime " + str(predicted_runtime)
			print("Point [{},{}] has predicted runtime -----> {}".format(f1, f2, predicted_runtime));
