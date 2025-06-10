#ifndef RAY_TRACE_CUH
#define RAY_TRACE_CUH
#include "structs.hpp"

__global__ void trace_rays(MeshPoint* mesh, Xyz<double>* deden,
		Xyz<double>* child, double* dist,
		size_t b_mem_num, size_t beamnum, Xyz<double>* ref_positions, double* TEMPoffsets,
		Crossing* crossings, size_t* turn, size_t rays_per_thread);
__device__ size_t launch_child_ray(MeshPoint* mesh, Xyz<double>* deden,
		Xyz<double> start_pos, Xyz<double> k0,
		Xyz<double>* traj, double* dist);
__device__ size_t launch_parent_ray(MeshPoint* mesh, Xyz<double>* deden,
		Xyz<double>* child1, double* dist1, size_t c1_size,
		Xyz<double>* child2, double* dist2, size_t c2_size,
		Xyz<double> pos, Xyz<double> k0, Crossing* crossings, size_t* turn, double TEMPintensity);
__device__ void calc_dk(Crossing* cross, Crossing* cross_next);
__device__ double get_k(MeshPoint* mesh, Xyz<double> point, Xyz<size_t> mesh_pos);
__device__ Xyz<double> interp_xyz(Xyz<double>* points, double* dist, size_t len, double xp);

template<typename T>
__device__ void swap(T& t1, T& t2) {
	T temp(t1);
	t1 = t2;
	t2 = temp;
}

#endif
