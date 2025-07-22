#ifndef CBET_CUH
#define CBET_CUH
#include "structs.hpp"

void post(MeshPoint* mesh, Crossing* crossings);
__global__ void calc_coupling_mults(MeshPoint* mesh, Crossing* crossings, RaystorePt* raystore, double* coupling_mults, size_t num_cmults);
__device__ double get_coupling_mult(MeshPoint* mesh, Crossing* cross, Crossing* raycross);
__global__ void get_cbet_gain(MeshPoint* mesh, Crossing* crossings,
		RaystorePt* raystore, double* w_mult_values,
		double* coupling_mults, size_t num_cmults);
__global__ void update_intensities(Crossing* crossings, RaystorePt* raystore,
		double* w_mult_values, double curr_max, double* updateconv);
__device__ double limit_energy(double i_prev, double i0, double mult_acc, double curr_max, double* max_change);

#endif
