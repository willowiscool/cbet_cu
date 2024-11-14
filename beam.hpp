#ifndef BEAM_H
#define BEAM_H
#include <cstddef>

// copied doc comment from rust:
/// Crossing struct stores
/// * x, z: the real-world coordinates of the crossing (c++: crossesx/z)
/// * boxesx, boxesz: the zone coordinates of the crossing (note: in c++, this is stored as
/// just boxes, where each element is a tuple)
/// * areaRatio: the ratio between parent and child ray
/// * dkx, dkz, dkmag: vectors of movement to next crossing, i guess. computed in main.cpp after
/// ray tracing runs, but figured it would be more efficient to compute them in the ray tracing
/// function. also, computed as dkx_new, dkz_new, dkmag_new in cpp impl.
/// * i_b: the intensity, calculated in the CBET stage.
/// * energy: the energy multiplier from absorption calculations, multiplied into the initial
/// intensity --- exp(kds)!
/// * absorption_coeff: next energy/current energy
// TODO: perhaps, remove some of these, because different values are used at different steps of computation, so to decrease memory footprint only use a certain type of crossing for a certain step and convert between them between steps.
struct Crossing {
	double x;
	double z;
	size_t boxesx;
	size_t boxesz;
	double area_ratio;
	double dkx;
	double dkz;
	double dkmag;
	double i_b;
	double energy;
	double absorption_coeff;
	double phase;
};

// CHANGE FROM RUST: crossings now part of Beam struct
// TODO: Maybe don't even need this, and can just have a function that derives these values from a ray index?
/// Ray struct stores:
/// * x0, z0: initial position
/// * cx0, cz0: initial position of child ray. the c++ impl. moves each ray in a single beam by
/// a child offset, so maybe it is better to have child offset x/z fields in the beam rather
/// than in the ray? But I'm defining it like this for now.
/// * kx0, kz0: initial velocity
struct Ray {
    double x0;
    double z0;
    double cx0;
    double cz0;
    double kx0;
    double kz0;
};

// TODO: Give beam a position and direction within it, and then make a general
// new beam function that takes position and drection and populates rays.
// For now, the functions that create the rays within the beams are hardcoded.
//
// * rays: you can tell
// * marked: a list of each spot on the mesh. Each list contains the crossings that pass through it, as a ray index and a crossing index.
// * raystore: similar to marked, but only one crossing per each spot on the mesh. It's a tuple with a boolean 'cause there may not be a crossing in that spot. (TODO - option type?)

struct Beam {
	Ray* rays;
	size_t nrays;
	Crossing* crossings;
	double intensity;
	// TODO ADD MARKED, RAYSTORE
};

Beam beam1();
Beam beam2();

#endif
