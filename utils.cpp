#include <cassert>
#include "consts.hpp"
#include "structs.hpp"
#include "utils.hpp"

using namespace std;

double mag(Xyz<double> pt) {
	return sqrt(pow(pt.x, 2) + pow(pt.y, 2) + pow(pt.z, 2));
}
Xyz<size_t> get_mesh_coords(MeshPoint* mesh, Xyz<double> xyz) {
	/*return get_mesh_coords_in_area(mesh, xyz,
		{0, 0, 0},
		{consts::NX-1, consts::NY-1, consts::NZ-1});*/
	// This implementation stolen from interp3D in Shuang's code.
	return Xyz<size_t>({ // thisx, thisy, thisz
		(size_t)floor((xyz.x - consts::XMIN) / consts::DX),
		(size_t)floor((xyz.y - consts::YMIN) / consts::DY),
		(size_t)floor((xyz.z - consts::ZMIN) / consts::DZ)
	});
	// NOTE: May overflow, or return invalid values!
	// Hopefully, when a trajectory is outside of the mesh,
	// that is checked BEFORE those values are used!
}
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
// used for interpolating child distance
Xyz<double> interp_xyz(Xyz<double>* points, double* dist, size_t len, double xp) {
	size_t low, high, mid;
	if (dist[0] <= dist[len-1]) {
		// x monotonically increase
		if (xp <= dist[0]) {
			return points[0];
		} else if (xp >= dist[len-1]) {
			return points[len-1];
		}

		low = 0;
		high = len - 1;
		mid = (low + high) >> 1;
		while (low < high - 1) {
			if (dist[mid] >= xp) {
				high = mid;
			} else {
				low = mid;
			}
			mid = (low + high) >> 1;
		}

		assert((xp >= dist[mid]) && (xp <= dist[mid+1]));
	} else {
		if (xp >= dist[0]) {
			return points[0];
		} else if (xp <= dist[len-1]) {
			return points[len-1];
		}

		low = 0;
		high = len - 1;
		mid = (low + high) >> 1;
		while (low < high - 1) {
			if (dist[mid] <= xp) {
				low = mid;
			} else {
				high = mid;
			}
			mid = (low + high) >> 1;
		}

		assert((xp <= dist[mid]) && (xp >= dist[mid+1]));
	}
	return {
		points[mid].x +
			((points[mid+1].x - points[mid].x)/(dist[mid+1]-dist[mid])*(xp-dist[mid])),
		points[mid].y +
			((points[mid+1].y - points[mid].y)/(dist[mid+1]-dist[mid])*(xp-dist[mid])),
		points[mid].z +
			((points[mid+1].z - points[mid].z)/(dist[mid+1]-dist[mid])*(xp-dist[mid]))
	};
}
