#ifndef RAY_TRACE_HPP
#define RAY_TRACE_HPP
#include "structs.hpp"

void ray_trace(MeshPoint* mesh, Crossing* crossings, size_t* turn);
size_t launch_child_ray(MeshPoint* mesh, Xyz<double>* deden,
		Xyz<double> start_pos, Xyz<double> k0,
		Xyz<double>* traj, double* dist);
size_t launch_parent_ray(MeshPoint* mesh, Xyz<double>* deden,
		Xyz<double>* child1, double* dist1, size_t c1_size,
		Xyz<double>* child2, double* dist2, size_t c2_size,
		Xyz<double> pos, Xyz<double> k0, Crossing* crossings, size_t* turn, double TEMPintensity);
void calc_dk(Crossing* cross, Crossing* cross_next);
double get_k(MeshPoint* mesh, Xyz<double> point, Xyz<size_t> mesh_pos);

#endif
