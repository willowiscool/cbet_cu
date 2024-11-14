#include <cstddef>
#include "beam.hpp"
#include "consts.hpp"

Beam beam1() {
	Ray* rays = new Ray[consts::NRAYS];
	Crossing* crossings = new Crossing[consts::NRAYS * consts::NCROSSINGS]();
	double dz = (consts::BEAM_MAX_Z-consts::BEAM_MIN_Z)/((double)consts::NRAYS-1);
	for (size_t i = 0; i < consts::NRAYS; i++) {
		double z0 = (double)i * dz + consts::BEAM_MIN_Z + consts::OFFSET1;
		rays[i] = Ray {
			consts::XMIN, // x0
			z0, // z0
			consts::XMIN, // cx0
			z0 + consts::CHILD_OFFSET, // cz0
			consts::DIR1[0], // kx0
			consts::DIR1[1], // kz0
		};
	}
	return Beam {
		rays,
		consts::NRAYS,
		crossings,
		consts::INTENSITY
	};
}

Beam beam2() {
	Ray* rays = new Ray[consts::NRAYS];
	Crossing* crossings = new Crossing[consts::NRAYS * consts::NCROSSINGS]();
	double dx = (consts::BEAM_MAX_Z-consts::BEAM_MIN_Z)/((double)consts::NRAYS-1);
	for (size_t i = 0; i < consts::NRAYS; i++) {
		double x0 = (double)i * dx + consts::BEAM_MIN_Z + consts::OFFSET2;
		rays[i] = Ray {
			x0, // x0
			consts::ZMIN+0.1e-4, // z0
			x0 + consts::CHILD_OFFSET, // cx0
			consts::ZMIN+0.1e-4, // cz0
			consts::DIR2[0], // kx0
			consts::DIR2[1], // kz0
		};
	}
	return Beam {
		rays,
		consts::NRAYS,
		crossings,
		consts::INTENSITY
	};
}
