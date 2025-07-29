#ifndef CBET_CUH
#define CBET_CUH
#include "structs.hpp"

struct CmultBounds {
	size_t start;
	size_t size;
};
struct Cmult {
	short beamnum;
	double coupling_mult;
};

void post(MeshPoint* mesh, Crossing* crossings);
__global__ void get_num_cmults(Crossing* crossings, RaystorePt* raystore, size_t* cuda_num_cmults);
__global__ void calc_coupling_mults(MeshPoint* mesh, Crossing* crossings, RaystorePt* raystore,
		CmultBounds* cmult_bounds, Cmult* cmults, size_t num_cmults_per_ray);
__device__ double get_coupling_mult(MeshPoint* mesh, Crossing* cross, Crossing* raycross);
__global__ void get_cbet_gain(MeshPoint* mesh, Crossing* crossings,
		RaystorePt* raystore, double* w_mult_values,
		CmultBounds* cmult_bounds, Cmult* cmults, size_t num_cmults_per_ray);
__global__ void update_intensities(Crossing* crossings, RaystorePt* raystore,
		double* w_mult_values, double curr_max, double* updateconv);
__device__ double limit_energy(double i_prev, double i0, double mult_acc, double curr_max, double* max_change);

template<typename T>
class Max {
	public:
		__device__ T operator()(const T &a, const T &b) {
			return a > b ? a : b;
		}
};

#endif
