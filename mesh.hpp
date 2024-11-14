#ifndef MESH_H
#define MESH_H

#include <cstddef>
#include <tuple>

using namespace std;

// Point stores all of the information at a given point in the mesh
//
// Instead of storing kib or permittivity, we store kib_multiplier and
// permittivity_multiplier, in order to prevent re-doing the same calculations
// many times
//
// let dielectric = 1 - eden / ncrit. then,
// kib_multiplier = 1e4 * kib * pow(eden / ncrit, 2) / sqrt(dielectric)
// permittivity_multiplier = permittivity * OMEGA / C_SPEED
struct Point {
	double x;
	double z;
	double eden;
	double machnum;
	double kib_multiplier;
	double permittivity_multiplier;
};

// Mesh stores the mesh
// * dx, dz are width and height of each zone
// * nx, nz are the number of zones in each direction
// * xmin, xmax, zmin, and zmax store the minimum real-world coordinates (inclusive)
struct Mesh {
	Point* points;
	double dx;
	double dz;
	size_t nx;
	size_t nz;
	double xmin;
	double xmax;
	double zmin;
	double zmax;

	static Mesh new_lin(double xmin, double xmax, size_t nx, double zmin, double zmax, size_t nz);

	Point& get(size_t x, size_t z);
	tuple<size_t, size_t> get_mesh_coords(double x, double z);
	tuple<size_t, size_t> get_mesh_coords_in_area(double x, double z, tuple<size_t, size_t> minpt, tuple<size_t, size_t> maxpt);
};

Mesh new_mesh();

#endif
