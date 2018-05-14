from pyhull.delaunay import DelaunayTri
import numpy as np

# f1,f2,f3,runtime(f1, f2,f3)
# TODO read from stdin
model_points_full = [[0,0,0,100],
					[32,32,32,41],
					[0,0,32,42],
					[0,32,0,43],
					[32,0,0,65],
					[9,32,32,55],
					[3,7,32,55],
					[19,32,16,55],
					[10,32,32,55],
					[1,3,23,55],
					[21,2,32,55],
					[1,7,12,55],
					[32,3,3,55],
					[0,32,32,55],
					[32,0,32,63],
					[32,32,0,39],
					[12,9,5,43],
					[21,3,3,35],
					[0,1,20,32]]

unknowns = [[16,8,24],
			[4,24,22],
			[24,4,4],
			[4,4,2]]

def get_runtime(point):
	for f1,f2,f3,runtime in model_points_full:
		if point[0] == f1 and point[1] == f2 and point[2] == f3:
			return runtime

# https://stackoverflow.com/questions/36270116/how-do-i-define-a-hyperplane-in-python-given-4-points-how-do-i-then-define-the
def calc_hyperplane(p1, p2, p3, p4):
   X=np.matrix([p1,p2,p3,p4])
   k=np.ones((4,1))
   a=np.matrix.dot(np.linalg.inv(X), k)
   print "equation is x * %s = 1" % a
   return a


# use hyperplane [a, b, c, d, e] to make prediction for point [f1, f2, f3]
def make_prediction(hyperplane, point):
#	a,b,c,d,e = hyperplane
	#return (d - a*point[0] - b*point[1])/c TODO
	return -1

# build model w/o runtime
model_points = [[f1,f2,f3] for [f1,f2,f3,f4] in model_points_full]

# do triangulation with model_points
# refer: https://pythonhosted.org/pyhull/
tri = DelaunayTri(model_points)
print("tri " + str(tri.simplices))

# check for unknowns belonging in each simplex
for simplex in tri.simplices:
	p1 = [simplex.coords[0].item(0), simplex.coords[0].item(1), simplex.coords[0].item(2), get_runtime([simplex.coords[0].item(0), simplex.coords[0].item(1), simplex.coords[0].item(2)])]
	p2 = [simplex.coords[1].item(0), simplex.coords[1].item(1), simplex.coords[1].item(2), get_runtime([simplex.coords[1].item(0), simplex.coords[1].item(1), simplex.coords[1].item(2)])]
	p3 = [simplex.coords[2].item(0), simplex.coords[2].item(1), simplex.coords[2].item(2), get_runtime([simplex.coords[2].item(0), simplex.coords[2].item(1), simplex.coords[2].item(2)])]
	p4 = [simplex.coords[3].item(0), simplex.coords[3].item(1), simplex.coords[3].item(2), get_runtime([simplex.coords[3].item(0), simplex.coords[3].item(1), simplex.coords[3].item(2)])]

	for f1,f2,f3 in unknowns:
		if simplex.in_simplex([f1, f2, f3]):
			print "Predicting runtime for " + str([f1, f2, f3])

			hyperplane = calc_hyperplane(p1, p2, p3, p4)
			predicted_runtime = make_prediction(hyperplane, [f1, f2, f3])
			print "predicted runtime " + str(predicted_runtime)
			print("Point {} has predicted runtime -----> {}".format([f1, f2, f3], predicted_runtime));
