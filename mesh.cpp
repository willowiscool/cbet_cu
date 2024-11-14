#include <cstddef>
#include <tuple>
#include <iostream>
#include <cmath>
#include "mesh.hpp"
#include "consts.hpp"

using namespace std;

// Function bodies for Mesh struct
// TODO: do I have to rename the arguments since they're the same as the values?
// Creates a new linear mesh. Points have their x, z values set and nothing else.
Mesh Mesh::new_lin(double xmin, double xmax, size_t nx, double zmin, double zmax, size_t nz) {
	double dx = (xmax - xmin)/((double)nx - 1.0);
	double dz = (zmax - zmin)/((double)nz - 1.0);
	Point* points = new Point[nx*nz];
	for (size_t meshx = 0; meshx < nx; meshx++) {
		for (size_t meshz = 0; meshz < nz; meshz++) {
			Point& p = points[(meshx*nz) + meshz];
			p.x = (double)meshx*dx + xmin;
			p.z = (double)meshz*dz + zmin;
		}
	}
	return Mesh {
		points,
		dx, dz, nx, nz, xmin, xmax, zmin, zmax,
	};
}

// Gets the point at mesh coordinates x, y (TODO: error check?)
Point& Mesh::get(size_t x, size_t z) {
	return points[(x*nz) + z];
}
// Gets the mesh coordinates of a point given its real-world coordinates
// Again, doesn't error check.
// Assumes a linear mesh.
// Does a linear search - binary search may be faster, but this code isn't
// used often at all.
tuple<size_t, size_t> Mesh::get_mesh_coords(double x, double z) {
	return get_mesh_coords_in_area(x, z, {0, 0}, {nx - 1, nz - 1});
}
// Does the same as above, only searching in the subsection of the mesh
// given by the coordinates minpt and maxpt, inclusive.
tuple<size_t, size_t> Mesh::get_mesh_coords_in_area(double x, double z, tuple<size_t, size_t> minpt, tuple<size_t, size_t> maxpt) {
	tuple<size_t, size_t> point = {0, 0};
	for (size_t xx = std::get<0>(minpt); xx <= std::get<0>(maxpt); xx++) {
		double px = get(xx, 0).x;
		if ((x - px <= (1.0 + 1.0e-10)*dx) &&
			(x - px >= -(0.0 + 1.0e-10)*dx)) {
			std::get<0>(point) = xx;
			break;
		}
	}
	for (size_t zz = std::get<1>(minpt); zz <= std::get<1>(maxpt); zz++) {
		double pz = get(0, zz).z;
		if ((z - pz <= (1.0 + 1.0e-10)*dz) &&
			(z - pz >= -(0.0 + 1.0e-10)*dz)) {
			std::get<1>(point) = zz;
			break;
		}
	}
	return point;
}

// Creates a new mesh using the consts defined in consts.h and here.
// It has a linear electron density gradient. Not implemented as Mesh::new
// since it's got too many constants and stuff.
Mesh new_mesh() {
	double ncrit = consts::NCRIT;
	double kib = 0.0;
	double xmax = consts::XMAX;
	double xmin = consts::XMIN;
	Mesh mesh = Mesh::new_lin(consts::XMIN, consts::XMAX, consts::NX, consts::ZMIN, consts::ZMAX, consts::NZ);
	for (size_t x = 0; x < consts::NX; x++) {
		for (size_t z = 0; z < consts::NZ; z++) {
			Point& p = mesh.get(x, z);
			double eden = std::max(0.0, ((0.4*ncrit-0.1*ncrit)/(xmax-xmin))*(p.x-xmin)+(0.1*ncrit));
			double sqrt_permittivity = std::sqrt(1.0 - (eden / ncrit));
			p.eden = eden;
			p.machnum = std::min(0.0, (((-1.8)-(-1.0))/(xmax-xmin))*(p.x-xmin))+(-1.0);
			p.kib_multiplier = 1e4 * kib * pow(eden / ncrit, 2) / sqrt_permittivity;
			p.permittivity_multiplier = std::max(sqrt_permittivity, 0.0) * consts::OMEGA / consts::C_SPEED;
		}
	}
	return mesh;
}
