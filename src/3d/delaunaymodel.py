from pyhull.delaunay import DelaunayTri
import numpy as np

class DelaunayModel:

	def __init__(self, points):
		# TODO pass in --> self.model_points = points
		# <[f1,f2],runtime(f1, f2)>
		self.model_points = points

	def get_runtime(self, point):
		for [f1,f2],runtime in self.model_points:
			if point[0] == f1 and point[1] == f2:
				return runtime

	# refer: http://kitchingroup.cheme.cmu.edu/blog/2015/01/18/Equation-of-a-plane-through-three-points/
	def calc_hyperplane(self, p1, p2, p3):
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
	def make_prediction(self, hyperplane, point):
		a,b,c,d = hyperplane
		return (d - a*point[0] - b*point[1])/c

	def construct_model(self):
		# build model w/o runtime
		model_points_features = [[f1,f2] for [[f1,f2],f3] in self.model_points]
		print "construct_model point count: " + str(len(self.model_points))
		print "Constructing model with points: " + str(self.model_points)

		# do initial triangulation with model_points
		# refer: https://pythonhosted.org/pyhull/
		self.tri = DelaunayTri(model_points_features)

	def predict(self, point):
		print "predict point=" + str(point)
		for simplex in self.tri.simplices:
			if simplex.in_simplex([point[0], point[1]]):
				p1 = [simplex.coords[0].item(0), simplex.coords[0].item(1), self.get_runtime([simplex.coords[0].item(0), simplex.coords[0].item(1)])]
				p2 = [simplex.coords[1].item(0), simplex.coords[1].item(1), self.get_runtime([simplex.coords[1].item(0), simplex.coords[1].item(1)])]
				p3 = [simplex.coords[2].item(0), simplex.coords[2].item(1), self.get_runtime([simplex.coords[2].item(0), simplex.coords[2].item(1)])]
			
				print "predicting for " + str(point) + " --> " + str([p1, p2, p3])
				hyperplane = self.calc_hyperplane(p1, p2, p3)
				predicted_runtime = self.make_prediction(hyperplane, point)
				print("Point {} has predicted runtime -----> {}".format(point, predicted_runtime));
				return predicted_runtime



