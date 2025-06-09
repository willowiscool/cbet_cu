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
	// double phase;
};

struct MeshPoint {
	Xyz<double> pt;
	Xyz<double> machnum;
	double eden;
	double kib;
	double kib_multiplier;
	// double permittivity_multiplier;
};

#endif
