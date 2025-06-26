#ifndef UTILS_HPP
#define UTILS_HPP
#include "structs.hpp"

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE //
#endif

template<typename T>
HOST_DEVICE
inline T* get_pt(T* arr, size_t x, size_t y, size_t z) {
	return arr + (x*consts::NY + y)*consts::NZ + z;
}
template<typename T>
HOST_DEVICE
inline T* get_pt(T* arr, Xyz<size_t> pt) {
	return get_pt(arr, pt.x, pt.y, pt.z);
}
HOST_DEVICE
double interp(double* y, double* x, size_t len, double xp);

#ifdef __CUDACC__
template<typename Func>
__device__ double interp3D(Xyz<double> point, Xyz<size_t> mesh_pos, Func field) {
	Xyz<double> pos = { // xp, yp, zp
		fmod((point.x - consts::XMIN), consts::DX) / consts::DX,
		fmod((point.y - consts::YMIN), consts::DY) / consts::DY,
		fmod((point.z - consts::ZMIN), consts::DZ) / consts::DZ
	};
	// move mesh pos back if at edge
	if (mesh_pos.x >= consts::NX - 1) {
		mesh_pos.x = consts::NX - 2;
		pos.x = 1.0;
	}
	if (mesh_pos.y >= consts::NY - 1) {
		mesh_pos.y = consts::NY - 2;
		pos.y = 1.0;
	}
	if (mesh_pos.z >= consts::NZ - 1) {
		mesh_pos.z = consts::NZ - 2;
		pos.z = 1.0;
	}
	double c00 =
		(1 - pos.x) * field(mesh_pos.x, mesh_pos.y, mesh_pos.z)
		+ pos.x * field(mesh_pos.x + 1, mesh_pos.y, mesh_pos.z);
	double c01 =
		(1 - pos.x) * field(mesh_pos.x, mesh_pos.y, mesh_pos.z + 1)
		+ pos.x * field(mesh_pos.x + 1, mesh_pos.y, mesh_pos.z + 1);
	double c10 =
		(1 - pos.x) * field(mesh_pos.x, mesh_pos.y + 1, mesh_pos.z)
		+ pos.x * field(mesh_pos.x + 1, mesh_pos.y + 1, mesh_pos.z);
	double c11 =
		(1 - pos.x) * field(mesh_pos.x, mesh_pos.y + 1, mesh_pos.z + 1)
		+ pos.x * field(mesh_pos.x + 1, mesh_pos.y + 1, mesh_pos.z + 1);

	double c0 = c00 * (1 - pos.y) + c10 * pos.y;
	double c1 = c01 * (1 - pos.y) + c11 * pos.y;
	return c0 * (1 - pos.z) + c1 * pos.z;
}

// Needed to move these in here so they wouldn't have to be relocated
// when CBET comes, this might? become an issue... but also might not
// since they're already here? anyway, lmk future willow
__device__ inline double mag(Xyz<double> pt) {
	return sqrt(pow(pt.x, 2) + pow(pt.y, 2) + pow(pt.z, 2));
}
__device__ inline Xyz<size_t> get_mesh_coords_in_area(MeshPoint* mesh, Xyz<double> xyz,
		Xyz<size_t> minpt, Xyz<size_t> maxpt) {
	Xyz<size_t> point = {0, 0, 0};
	for (size_t xx = minpt.x; xx <= maxpt.x; xx++) {
		double px = get_pt(mesh, xx, 0, 0)->pt.x;
		if ((xyz.x - px <= (1.0 + 1.0e-20)*consts::DX) &&
			(xyz.x - px >= -(0.0 + 1.0e-20)*consts::DX)) {
			point.x = xx;
			break;
		}
	}
	for (size_t yy = minpt.y; yy <= maxpt.y; yy++) {
		double py = get_pt(mesh, 0, yy, 0)->pt.y;
		if ((xyz.y - py <= (1.0 + 1.0e-20)*consts::DY) &&
			(xyz.y - py >= -(0.0 + 1.0e-20)*consts::DY)) {
			point.y = yy;
			break;
		}
	}
	for (size_t zz = minpt.z; zz <= maxpt.z; zz++) {
		double pz = get_pt(mesh, 0, 0, zz)->pt.z;
		if ((xyz.z - pz <= (1.0 + 1.0e-20)*consts::DZ) &&
			(xyz.z - pz >= -(0.0 + 1.0e-20)*consts::DZ)) {
			point.z = zz;
			break;
		}
	}
	return point;
}
__device__ inline Xyz<size_t> get_mesh_coords(MeshPoint* mesh, Xyz<double> xyz) {
	return get_mesh_coords_in_area(mesh, xyz,
		{0, 0, 0},
		{consts::NX-1, consts::NY-1, consts::NZ-1});
}
__device__ Xyz<size_t> get_mesh_coords(MeshPoint* mesh, Xyz<double> xyz);
#endif

#endif
