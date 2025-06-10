#include <cassert>
#include "consts.hpp"
#include "structs.hpp"
#include "utils.cuh"

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE //
#endif

using namespace std;

/* Old implementation that searches for mesh zone
 * Useful if mesh isn't linearly spaced.
Xyz<size_t> get_mesh_coords_in_area(MeshPoint* mesh, Xyz<double> xyz,
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
}*/

// panics instead of erroring... or undefined here in cpp?
HOST_DEVICE
double interp(double* y, double* x, size_t len, double xp) {
	if (x[0] <= x[len-1]) {
		// x monotonically increase
		if (xp <= x[0]) {
			return y[0];
		} else if (xp >= x[len-1]) {
			return y[len-1];
		}

		size_t low = 0;
		size_t high = len - 1;
		size_t mid = (low + high) >> 1;
		while (low < high - 1) {
			if (x[mid] >= xp) {
				high = mid;
			} else {
				low = mid;
			}
			mid = (low + high) >> 1;
		}

		assert((xp >= x[mid]) && (xp <= x[mid+1]));
		return y[mid] + ((y[mid+1] - y[mid]) / (x[mid+1] - x[mid]) * (xp - x[mid]));
	} else {
		if (xp >= x[0]) {
			return y[0];
		} else if (xp <= x[len-1]) {
			return y[len-1];
		}

		size_t low = 0;
		size_t high = len - 1;
		size_t mid = (low + high) >> 1;
		while (low < high - 1) {
			if (x[mid] <= xp) {
				low = mid;
			} else {
				high = mid;
			}
			mid = (low + high) >> 1;
		}

		assert((xp <= x[mid]) && (xp >= x[mid+1]));
		return y[mid] + ((y[mid+1]-y[mid])/(x[mid+1]-x[mid]) * (xp-x[mid]));
	}
}
