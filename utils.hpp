#ifndef UTILS_HPP
#define UTILS_HPP
#include "structs.hpp"

template<typename T>
inline T* get_pt(T* arr, size_t x, size_t y, size_t z) {
	return arr + (x*consts::NY + y)*consts::NZ + z;
}
template<typename T>
inline T* get_pt(T* arr, Xyz<size_t> pt) {
	return get_pt(arr, pt.x, pt.y, pt.z);
}
template<typename Func>
double interp3D(Xyz<double> point, Xyz<size_t> mesh_pos, Func field) {
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

double mag(Xyz<double> pt);
double interp(double* y, double* x, size_t len, double xp);
Xyz<double> interp_xyz(Xyz<double>* points, double* dist, size_t len, double xp);
//Xyz<size_t> get_mesh_coords_in_area(MeshPoint* mesh, Xyz<double> xyz,
//		Xyz<size_t> minpt, Xyz<size_t> maxpt);
Xyz<size_t> get_mesh_coords(MeshPoint* mesh, Xyz<double> xyz);

#endif
