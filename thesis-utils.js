// init pattern: https://stackoverflow.com/questions/2190801/passing-parameters-to-javascript-files
var ThesisUtils = ThesisUtils || (function(){
    var _args = {}; // private
	
	var voronoi;
	var delaunay_triangles = [];

    return {
        init : function(Args) {
            _args = Args;
        },
		voronoi : this.voronoi,
		delaunay_triangles : this.delaunay_triangles,
		draw_plot : function() {
			/*
				LOTS OF INITIAL D3.JS HELP HERE FROM:
				https://www.visualcinnamon.com/2015/07/voronoi.html
				http://bl.ocks.org/d3noob/38744a17f9c0141bcd04
			*/

			// Set the dimensions of the canvas / graph
			var margin = {top: 50, right: 30, bottom: 50, left: 30},
				width = 600 - margin.left - margin.right,
				height = 600 - margin.top - margin.bottom;

			// Set the ranges
			var xScale = d3.scale.linear().range([0, width]);
			var yScale = d3.scale.linear().range([height, 0]);

			// Define the axes
			var xAxis = d3.svg.axis().scale(xScale)
				.orient("bottom").ticks(5);

			var yAxis = d3.svg.axis().scale(yScale)
				.orient("left").ticks(5);

			var notifier = d3.select("body")
				.append("div")
					.attr("class", "notifier");

			// Adds the svg canvas
			var svg = d3.select("body")
				.append("svg")
					.attr("width", width + margin.left + margin.right)
					.attr("height", height + margin.top + margin.bottom)
				.append("g")
					.attr("transform", 
						 "translate(" + margin.left + "," + margin.top + ")")
				
			// Scale the range of the data
			xScale.domain([0, 64]);
			yScale.domain([0, 16]);

			var showTooltip = function(d) {
				notifier.html("<h2>Last Point:</h2><h2>" + JSON.stringify(d) + "</h2>");
			}

			// Add the scatterplot of input data
			svg.selectAll(".dot")
				.data(_args.data)
				.enter().append("circle")
				.attr("class", "dot")
				.attr("r", 5)
				.attr("cx", function(d) { return xScale(d.vmem); })
				.attr("cy", function(d) { return yScale(d.vcores); })
				.on("mouseover", showTooltip)
				//.on("mouseout", removeTooltip);

			if(_args.do_triangulation) {
				chart_title = "Delaunay Trianglation of " + _args.chart_title;
			}

			if(_args.do_voronoi) {
				chart_title = "Voronoi Plot of " + chart_title;
			}

			svg.append("text")
				.attr("x", (width / 2))             
				.attr("y", 0 - (margin.top / 2))
				.attr("text-anchor", "middle")  
				.style("font-size", "16px") 
				.text(chart_title);

			// Add the X Axis
		   svg.append("g")
				.attr("class", "x axis")
				.attr("transform", "translate(0," + height + ")")
				.call(xAxis);

			// X-Axis Label
			svg.append("text")
				.attr("x", (width / 2))             
				.attr("y", height + margin.bottom - 10)
				.attr("text-anchor", "middle")  
				.style("font-size", "16px") 
				.text("Virtual Memory");
			
			// Add the Y Axis
			svg.append("g")
				.attr("class", "y axis")
				.call(yAxis);

			// Y-Axis Label
			svg.append("text")
				.attr("transform", "rotate(-90)")
				.attr("y", 0 - margin.left)
				.attr("x",0 - (height / 2))
				.attr("dy", "1em")
				.style("font-size", "16px")
				.style("text-anchor", "middle")
				.text("Virtual Cores");

			// address bug in d3-voronoi lib which results in triangles not being created if a lot of points happen to be collinear (in a line)
			// add a slight "jitter" to each point, which averages out to 0
			// https://github.com/d3/d3-voronoi/issues/12
			this.voronoi = d3.geom.voronoi()
				.x(function(d) { return xScale(d.vmem + Math.random() - 0.5) })
				.y(function(d) { return yScale(d.vcores + Math.random() - 0.5) })
				.clipExtent([[0, 0], [width, height]]);

			// get delaunay triangle coordinates for dataset
			this.delaunay_triangles = this.voronoi.triangles(data).map(function(triangle) {
				var a = [triangle[0].vmem, triangle[0].vcores, triangle[0].time];
				var b = [triangle[1].vmem, triangle[1].vcores, triangle[1].time];
				var c = [triangle[2].vmem, triangle[2].vcores, triangle[2].time];

				// from each triangle, compute the plane passing through the three points of it
				var plane = ThesisUtils.calculate_plane(a, b, c);

				// attach a function to each triangle, which gives the value of this plane with any (vmem, vcore) configuration
				triangle.planeValue = function(vmem, vcores) {
					return (plane[3] - plane[1]*vcores - plane[0]*vmem)/plane[2];		
				};

				// attach function to check if this triangle contains the given vertex
				triangle.contains_vertex = function(x, y) {
					return ((triangle[0].vmem == x && triangle[0].vcores == y) || (triangle[1].vmem == x && triangle[1].vcores == y) || (triangle[2].vmem == x && triangle[2].vcores == y));	
				};

				return triangle;
			});

			if(_args.do_unknowns) {
				svg.selectAll(".dot-unknown")
					.data(unknowns)
					.enter().append("circle")
					.attr("class", "dot-unknown")
					.attr("r", 10)
					.attr("cx", function(d) { return xScale(d.vmem); })
					.attr("cy", function(d) { return yScale(d.vcores); });
		
				var unknowns_container = d3.select("body")
					.append("div")
						.attr("class", "unknowns-container")
					.html("<h2>Unknowns</h2>")

				unknowns.forEach(function(unknown) {
					unknowns_container.append("p").html("<h3>" + JSON.stringify(unknown) + " --> " + JSON.stringify(ThesisUtils.get_triangle_for_point(unknown.vmem, unknown.vcores)) + " </h3>");
				});
			}

			//Create the Voronoi grid
			if(_args.do_voronoi) {
				svg.selectAll(".polygon")
					.data(voronoi(data))
					.enter().append("path")
					.attr("class", "polygon")
					.attr("d", function(d, i) { return "M" + d.join("L") + "Z"; })
					.datum(function(d, i) { return d.point; })
			}

			// add the Delaunay triangulation
			if(_args.do_triangulation) {
				svg.selectAll(".triangle-link")
					.data(this.delaunay_triangles)
					.enter().append("polygon")
					.attr("class", "triangle-link")
					.attr("points",function(d) {
						return d.map(function(d) {
							return [xScale(d.vmem),yScale(d.vcores)].join(",");
					}).join(" ");
				});
			}
		},
		calculate_plane : function(p, q, r) {
			console.log("calculate_plane: p=" + p + "q=" + q + "r=" + r);
			/*
			 * Calculate the plane that rests on points p, q, and r
			 */
			if(!p || p.length !== 3 || !q || q.length !== 3 || !r || r.length !== 3) {
				console.error("Invalid input to calculate_plane");
				return;
			}

			// get two vectors fully in the plane
			var pq = [(q[0] - p[0]), (q[1] - p[1]), (q[2] - p[2])];
			var pr = [(r[0] - p[0]), (r[1] - p[1]), (r[2] - p[2])];

			// cross to get orthogonal vector to plane
			// <a, b, c,> x <d, e, f> = (bf-ce)i - (af-cd)j + (ar-bd)k = <(bf-ce), -(af-cd), (ae-bd)>
			var a = pq[0], b = pq[1], c = pq[2]
				d = pr[0], e = pr[1], f = pr[2],
				pq_cross_pr = [(b*f - c*e), -1 * (a*f - c*d), (a*e - b*d)]

			a = pq_cross_pr[0];
			b = pq_cross_pr[1];
			c = pq_cross_pr[2];

			// plug in point p to finally get d
			d = a*p[0] + b*p[1] + c*p[2]; 

			// put into scalar form
			/*console.log("Plane equation is: {0}x + {1}y + {2}z = {3}"
				.replace("{0}", a)
				.replace("{1}", b)
				.replace("{2}", c)
				.replace("{3}", d));*/

			console.log("Solved for Z (time): z = ({3} - {1}y - {0}x)/{2}"
				.replace("{0}", a)
				.replace("{1}", b)
				.replace("{2}", c)
				.replace("{3}", d));

			return [a, b, c, d];
		},
		line_contains_point : function(px, py, x1, y1, x2, y2) {
			// check if (px, py) is on the line segment (x1, y1) | --- | (x2, y2)
			// the line contains (px, py) if the distance from (px, py) to (x1, y1) and (x2, y2) is equal to the length of the entire line
			return (this.euclidean_dist(px, py, x1, y1) + this.euclidean_dist(px, py, x2, y2) == this.euclidean_dist(x1, y1, x2, y2));
		},
		triangle_contains_point : function(px, py, x1, y1, x2, y2, x3, y3) {
			/*console.log("triangle_contains_point({0}, {1}, {2}, {3}, {4}, {5}, {6}, {7})"
				.replace("{0}", px)
				.replace("{1}", py)
				.replace("{2}", x1)
				.replace("{3}", y1)
				.replace("{4}", x2)
				.replace("{5}", y2)
				.replace("{6}", x3)
				.replace("{7}", y3)); */
			// imagine an arbitrary triangle made with points a=(x1, y1), b=(x2, y2) and c=(x3, y3)
			// the point in question is p=(px, py) = (vmem, vcores)
			// to determine if a point lies within the triangle, take the following cross products and check the sign of each determinant:
				// ap x ab, bp x bc and cp x ca
			// if the sign is the same for all three, the point lies within the triangle

			var ap_ab_det = ((px-x1)*(y2-y1) - (py-y1)*(x2-x1));
			var bp_bc_det = ((px-x2)*(y3-y2) - (py-y2)*(x3-x2));
			var cp_ca_det = ((px-x3)*(y1-y3) - (py-y3)*(x1-x3));

			// we'll count it if it's on the edge
			var is_on_edge = this.line_contains_point(px, py, x1, y1, x2, y2) ||
				this.line_contains_point(px, py, x1, y1, x3, y3) ||
				this.line_contains_point(px, py, x2, y2, x3, y3);

			return (ap_ab_det > 0 && bp_bc_det > 0 && cp_ca_det > 0) || (ap_ab_det < 0 && bp_bc_det < 0 && cp_ca_det < 0) || is_on_edge;
		},
		get_triangle_for_point : function(vmem, vcores) {
			// loop thru delaunay_triangles and find the one containing vmem and vcords
			var px = vmem, py = vcores;
			var matching_triangle;
			
			for(var i = 0; i < this.delaunay_triangles.length; i++) {
				var triangle = this.delaunay_triangles[i];
				var x1 = triangle[0].vmem, y1 = triangle[0].vcores,
					x2 = triangle[1].vmem, y2 = triangle[1].vcores,
					x3 = triangle[2].vmem, y3 = triangle[2].vcores;

				if(this.triangle_contains_point(px, py, x1, y1, x2, y2, x3, y3)) {
					matching_triangle = triangle;
				}
			}

			// if no triangle found, it means the point is outside the convex hull / triangles
			// in this case, find the nearest two points 
			if(!matching_triangle) {
				console.log("No triangle found for point <" + vmem + ", " + vcores + ">, tried following triangles: " + this.delaunay_triangles);

				var nearest_2_points = this.find_nearest_points(px, py, 2);
				var p1 = nearest_2_points[0].to_point,
					p2 = nearest_2_points[1].to_point;


				// TODO: this isn't working quite right

				// search thru delaunay_triangles for one with these two points
				// we can assume (based upon context) there will be only 1 triangle with these two points
				for(var i = 0; i < this.delaunay_triangles.length; i++) {
					var next_triangle = this.delaunay_triangles[i];
					if(next_triangle.contains_vertex(p1[0], p1[1]) && next_triangle.contains_vertex(p2[0], p2[1])) {
						matching_triangle = next_triangle;
						break;
					}
				};
			}
	
			return matching_triangle; 
		},
		triangle_contains_vertex(vx, xy, x1, y1, x2, y2, x3, y3) {
			// check if one of the verticies in the triangle (x1, y1), (x2, y2), (x3, y3) equals (vx, xy)
			return (vx == x1 && vy == y1) || (vx == x2 && vy == y2) || (vx == x3 && vy == y3);
		},
		find_nearest_points(px, py, n) {
			// find n nearest points from (px, py)
			// find distance to all data points from (px, py)
			// sort distances and take n
			
			return _args.data.map(function(point) {
				var point = _args.data[0];
				var point_x = point.vmem;
				var point_y = point.vcores;

				return {
					to_point: [point_x, point_y],
					distance:  ThesisUtils.euclidean_dist(px, py, point_x, point_y)
				};
			}).sort(function(a, b){ return a.distance - b.distance; }).slice(0, n);
		},
		euclidean_dist: function(x1, y1, x2, y2) {
			// calculate euclidean distance between two points
			return Math.sqrt(Math.pow(Math.abs(x2 - x1), 2) + Math.pow(Math.abs(y2 - y1), 2));
		},
		triangle_area: function(x1, y1, x2, y2, x3, y3) {
			// calculate area of triangle by heron's formula: https://en.wikipedia.org/wiki/Heron%27s_formula
			var sidelen_a = this.euclidean_dist(x1, y1, x2, y2);
			var sidelen_b = this.euclidean_dist(x1, y1, x3, y3);
			var sidelen_c = this.euclidean_dist(x2, y2, x3, y3);
			var semi_perimeter = (sidelen_a + sidelen_b + sidelen_c) / 2;
			return Math.sqrt(semi_perimeter * (semi_perimeter - sidelen_a) * (semi_perimeter - sidelen_b) * (semi_perimeter - sidelen_c));
		}
    };
}());
