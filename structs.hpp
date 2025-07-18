#ifndef STRUCTS_HPP
#define STRUCTS_HPP
#include <cstddef>

template <typename T>
struct Xyz {
	T x;
	T y;
	T z;
};

// This is the crossing for CUDA raytrace, so the following are not present:
// dkx/y/z, dkmag, i_b, absorption_coeff
struct Crossing {
	Xyz<double> pt;
	Xyz<size_t> boxes;
	Xyz<double> dk;
	double dkmag;
	double area_ratio;
	double kds;
	double i_b;
	double absorption_data;
	double absorb_coeff;
	// double phase;
};

struct MeshPoint {
	Xyz<double> pt;
	Xyz<double> machnum;
	double eden;
	double kib;
	double dielectric;
	// double permittivity_multiplier;
};

struct RaystorePt {
	size_t raynum;
	size_t cnum;
};

// used in CBET
struct CbetCrosses {
	size_t b_num;
	size_t r_num;
	size_t c_num;
	size_t c_num_next;
	double coupling_mult;
};
struct CbetCrossing {
	double intensity;
	double absorption_coeff;
};

#endif
