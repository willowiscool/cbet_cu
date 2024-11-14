#include <cstddef>
#include <cassert>
#include <iostream>
#include "consts.hpp"
#include "beam.hpp"
#include "mesh.hpp"
#include "ray_trace.hpp"

/// The ray_trace function populates the crossings of the rays in the beam array.
/// It relies heavily on constants defined in consts.rs - TODO: stop doing that
size_t ray_trace(Mesh& mesh, Beam* beams, size_t nbeams) {
	// first, calculate dedendz, dedendx
	// deden = derivative of eden
	double* dedendx = new double[mesh.nx*mesh.nz];
	double* dedendz = new double[mesh.nx*mesh.nz];

	// skip points on outer edges of mesh
	for (size_t x = 0; x < mesh.nx - 1; x++) {
		for (size_t z = 0; z < mesh.nz - 1; z++) {
			Point& p = mesh.get(x, z);
			if (x != 0) {
				Point& prev_point_x = mesh.get(x-1, z);
				dedendx[(x*mesh.nz)+z] = (p.eden - prev_point_x.eden)/(p.x - prev_point_x.x);
			}
			if (z != 0) {
				Point& prev_point_z = mesh.get(x, z-1);
				dedendz[(x*mesh.nz)+z] = (p.eden - prev_point_z.eden)/(p.z - prev_point_z.z);
			}
		}
	}
	// fill in outer points
	// ROW - i is the column in the row that's being modified
	for (size_t i = 0; i < mesh.nz; i++) {
		// first row: x = 0, fill in xes only
		// mesh.nz+i is point at (1, i)
		dedendx[i] = dedendx[mesh.nz+i];
		// fill in x = mesh.nx - 1
		dedendx[(mesh.nx-1)*mesh.nz+i] = dedendx[(mesh.nx-2)*mesh.nz+i];
		dedendz[(mesh.nx-1)*mesh.nz+i] = dedendz[(mesh.nx-2)*mesh.nz+i];
	}
	// COL - i is the row in the column that's being modified
	for (size_t i = 0; i < mesh.nx; i++) {
		// first col: z = 0, fill in zs only
		dedendz[i*mesh.nz] = dedendz[i*mesh.nz+1];
		// fill in z = mesh.nz - 1
		dedendx[i*mesh.nz + (mesh.nz-1)] = dedendx[i*mesh.nz + (mesh.nz-2)];
		dedendz[i*mesh.nz + (mesh.nz-1)] = dedendz[i*mesh.nz + (mesh.nz-2)];
	}
	// in loop - code loop
	// might these be too big? TODO: investigate what size they should be
	// can't use vectors for GPU code.
	double* x = new double[consts::NT];
	double* z = new double[consts::NT];
	double* dist = new double[consts::NT];
	size_t ncrossings = 0;
	for (size_t beamnum = 0; beamnum < nbeams; beamnum++) {
		Beam& beam = beams[beamnum];
		for (size_t raynum = 0; raynum < beam.nrays; raynum++) {
			size_t child_size = launch_child_ray(beam.rays[raynum], mesh, dedendx, dedendz, x, z, dist);
			size_t cnum = launch_parent_ray(
				beam.rays[raynum], beam.crossings + (consts::NCROSSINGS * raynum), mesh,
				dedendx, dedendz,
				x, z, dist, child_size,
				raynum);
			ncrossings = max(cnum, ncrossings);
		}
	}

	delete[] dedendx;
	delete[] dedendz;
	delete[] x;
	delete[] z;
	delete[] dist;

	return ncrossings;
}

/// Launches the parent ray, adding the corresponding crossings.
/// More detailed comments may be found in launch_child_ray
/// TODO: Implement marked
size_t launch_parent_ray(Ray& ray, Crossing* crossings, Mesh& mesh,
	double* dedendx, double* dedendz,
	double* childx, double* childz, double* distance, size_t child_size,
	size_t raynum) {
	size_t cnum = 0;

	double x = ray.x0;
	double z = ray.z0;

	// convert child coordinates into a set of distances
	// TODO-PRIORITY: do in place in launch_child_ray?
	/*double* distance = new double[child_size];
	distance[0] = 0.0;
	for (size_t i = 1; i < child_size; i++) {
		distance[i] = distance[i-1] + sqrt(pow(childx[i] - childx[i-1], 2) + pow(childz[i] - childz[i-1], 2));
	}*/

	// compute init_area
	double init_diff_x = ray.x0 - childx[0];
	double init_diff_z = ray.z0 - childz[0];
	double init_diff_mag = sqrt(pow(init_diff_x, 2) + pow(init_diff_z, 2));
	double init_proj_coeff = abs(ray.kx0 * (init_diff_z/init_diff_mag) - ray.kz0 * (init_diff_x/init_diff_mag));
	double init_area = init_diff_mag*init_proj_coeff;

	auto [meshx, meshz] = mesh.get_mesh_coords(ray.x0, ray.z0);

	double k = get_k(mesh, ray.x0);
	double knorm = sqrt(pow(ray.kx0, 2) + pow(ray.kz0, 2));

	double vx = pow(consts::C_SPEED, 2) * ((ray.kx0/knorm) * k) / consts::OMEGA;
	double vz = pow(consts::C_SPEED, 2) * ((ray.kz0/knorm) * k) / consts::OMEGA;

	double kds = 0;
	double phase = 0;

	double curr_dist = 0;

	// TEMPORARY
	double offset = consts::BEAM_MIN_Z + ((double)raynum * (consts::BEAM_MAX_Z-consts::BEAM_MIN_Z)/(consts::NRAYS-1));
	double intensity = (consts::INTENSITY/1e14)*exp(-2.0*pow(abs(offset/2e-4),4));
	// END TEMPORARY
	
	for (size_t _ = 1; _ < consts::NT; _++) {
		// 1. Calculate velocity and position at current timestamp 
		double my_dedendx =
			meshx == mesh.nx - 1 ? dedendx[meshx*mesh.nz + meshz] :
			dedendx[meshx*mesh.nz + meshz] + (
				(dedendx[(meshx+1)*mesh.nz + meshz] - dedendx[meshx*mesh.nz + meshz])
				/ (mesh.get(meshx+1, meshz).x - mesh.get(meshx, meshz).x)
				* (x - mesh.get(meshx, meshz).x)
			);

		double prev_vx = vx;
		double prev_vz = vz;
		vx = vx - pow(consts::C_SPEED, 2) / (2.0 * consts::NCRIT) * my_dedendx * consts::DT;
		vz = vz - pow(consts::C_SPEED, 2) / (2.0 * consts::NCRIT) * dedendz[meshx*mesh.nz + meshz] * consts::DT;
		double prev_x = x;
		double prev_z = z;
		x = prev_x + vx * consts::DT;
		z = prev_z + vz * consts::DT;

		// 2. update meshx, meshz
		size_t prev_meshx = meshx;
		size_t prev_meshz = meshz;
		tuple<size_t, size_t> mcoords = mesh.get_mesh_coords_in_area(
			x, z,
			{ meshx == 0 ? 0 : meshx-1, meshz == 0 ? 0 : meshz-1 },
			{ min(mesh.nx-1, meshx+1), min(mesh.nz-1, meshz+1) }
		);
		meshx = get<0>(mcoords);
		meshz = get<1>(mcoords);
		Point& meshpt = mesh.get(meshx, meshz);

		// needs to borrow a ton of variables from its environment
		// ***instead of returning a crossing, it modifies the one at the crossings index, and then increases that index***
		auto new_crossing = [&] (double x, double z, double frac) {
			double distance_to_crossing = sqrt(pow(x - prev_x, 2) + pow(z - prev_z, 2));
			crossings[cnum].x = x;
			crossings[cnum].z = z;
			crossings[cnum].boxesx = meshx;
			crossings[cnum].boxesz = meshz;
			// Compute area ratio
			double childxp = interp(childx, distance, child_size, curr_dist + distance_to_crossing);
			double childzp = interp(childz, distance, child_size, curr_dist + distance_to_crossing);

			double diff_x = x - childxp;
			double diff_z = z - childzp;
			double diff_mag = sqrt(pow(diff_x, 2) + pow(diff_z, 2));

			double interpkx = frac*prev_vx + (1.0-frac)*vx;
			double interpkz = frac*prev_vz + (1.0-frac)*vz;
			double interpk_mag = sqrt(pow(interpkx, 2) + pow(interpkz, 2));

			double proj_coeff = abs((interpkx/interpk_mag) * (diff_z/diff_mag) - (interpkz/interpk_mag) * (diff_x/diff_mag));

			crossings[cnum].area_ratio = diff_mag*proj_coeff/init_area;
			crossings[cnum].energy = exp(kds - distance_to_crossing * meshpt.kib_multiplier);
			crossings[cnum].phase = phase + distance_to_crossing * meshpt.permittivity_multiplier;
			// all other values left at zero: dkx, dkz, dkmag, i_b, absorption_coeff
			// TEMP:
			crossings[cnum].i_b = intensity;
			cnum++;
		};

		double lastx = 10000;
		double lastz = 10000;
		bool is_cross_x = false;
		bool is_cross_z = false;

		// NOTE (IMPORTANT): This code might assume a linear mesh? and ***assumes there is no more than one crossing per dimension in any given time step***
		if (meshx != prev_meshx) {
			// get real x coord of crossing
			double currx = mesh.get(meshx, 0).x;
			if (!((x > currx && prev_x <= currx) || (x < currx && prev_x >= currx))) {
				currx = mesh.get(prev_meshx, 0).x;
			}

			// find the z point of intersection
			// crossx is a z coordinate!
			double crossx = prev_z + ((z - prev_z) / (x - prev_x) * (currx - prev_x)); // ***changed***
			double frac = (currx - prev_x) / (x - prev_x);
			assert((frac >= 0.0) && (frac <= 1.0));
			if (abs(crossx - lastz) > 1.0e-12) {
				new_crossing(currx, crossx, frac);
				// TODO marked
				is_cross_x = true;
				lastx = currx;
			}
		}
		if (meshz != prev_meshz) {
			double currz = mesh.get(0, meshz).z;
			if (!((z > currz && prev_z <= currz) || (z < currz && prev_z >= currz))) {
				currz = mesh.get(0, prev_meshz).z;
			}

			double crossz = prev_x + ((x - prev_x) / (z - prev_z)) * (currz - prev_z);
			double frac = (currz - prev_z) / (z - prev_z);
			assert((frac >= 0.0) && (frac <= 1.0));

			if (abs(crossz - lastx) > 1.0e-12) {
				new_crossing(crossz, currz, frac);
				// TODO marked

				is_cross_z = true;
				lastz = currz;
			}
		}

		// swap if out of order. then, calculate dkx, dkz, dkmag for prev. crossings
		if (is_cross_x && is_cross_z) {
			if ((x - prev_x) * (crossings[cnum-1].x - crossings[cnum-2].x) < 0.0) {
				swap(crossings[cnum-1], crossings[cnum-2]);
			}
			if (cnum > 2) {
				calc_dk(crossings[cnum-3], crossings[cnum-2]);
			}
			calc_dk(crossings[cnum-2], crossings[cnum-1]);
		} else if ((is_cross_x || is_cross_z) && cnum > 1) {
			calc_dk(crossings[cnum-2], crossings[cnum-1]);
		}
		double distance_travelled = sqrt(pow(x - prev_x, 2) + pow(z - prev_z, 2));
		curr_dist += distance_travelled;
		kds -= distance_travelled * meshpt.kib_multiplier;
		phase += distance_travelled * meshpt.permittivity_multiplier;
		if (x < mesh.xmin || x > mesh.xmax || z < mesh.zmin || z > mesh.zmax) break;
	}
	
	return cnum;
}

void calc_dk(Crossing& cross, Crossing& cross_next) {
	double dkx = cross_next.x - cross.x;
	double dkz = cross_next.z - cross.z;
	double dkmag = sqrt(pow(dkx, 2) + pow(dkz, 2));
	// normalize
	double dkx_new = dkx/dkmag;
	double dkz_new = dkz/dkmag;
	double dkmag_new = dkmag*10000.0;

	cross.dkx = dkx_new;
	cross.dkz = dkz_new;
	cross.dkmag = dkmag_new;
}

/// Launch the child ray. Modifies the x, z vectors, and returns the final length reached
size_t launch_child_ray(Ray& ray, Mesh& mesh, double* dedendx, double* dedendz, double* x, double* z, double* dist) {
	size_t ind = 0;

	x[ind] = ray.cx0;
	z[ind] = ray.cz0;
	dist[ind] = 0.0;
	ind++;

	// structured binding - c++17
	auto [meshx, meshz] = mesh.get_mesh_coords(ray.cx0, ray.cz0);
	/*size_t meshx = get<0>(mcoords);
	size_t meshz = get<1>(mcoords);*/

	double k = get_k(mesh, ray.cx0);
	double knorm = sqrt(pow(ray.kx0, 2) + pow(ray.kz0, 2));
	double vx = pow(consts::C_SPEED, 2) * ((ray.kx0 / knorm) * k) / consts::OMEGA;
	double vz = pow(consts::C_SPEED, 2) * ((ray.kz0 / knorm) * k) / consts::OMEGA;

	while (ind < consts::NT) {
		// 1. Calculate velocity and position at current timestamp
		double my_dedendx =
			meshx == mesh.nx - 1 ? dedendx[meshx*mesh.nz + meshz] :
			dedendx[meshx*mesh.nz + meshz] + (
				(dedendx[(meshx+1)*mesh.nz + meshz] - dedendx[meshx*mesh.nz + meshz])
				/ (mesh.get(meshx+1, meshz).x - mesh.get(meshx, meshz).x)
				* (x[ind-1] - mesh.get(meshx, meshz).x)
			);
		// my_dedendx is computed but not my_dedendz. This is becuase in current test cases, given the linear electron gradient, dedendz is consistent across a row.

		vx = vx - pow(consts::C_SPEED, 2) / (2.0 * consts::NCRIT) * my_dedendx * consts::DT;
		vz = vz - pow(consts::C_SPEED, 2) / (2.0 * consts::NCRIT) * dedendz[meshx*mesh.nz + meshz] * consts::DT;

		x[ind] = x[ind-1] + vx * consts::DT;
		z[ind] = z[ind-1] + vz * consts::DT;
		dist[ind] = dist[ind-1] + sqrt(pow(x[ind] - x[ind-1], 2) + pow(z[ind] - z[ind-1], 2));

		// 2. update meshx, meshz (for vel calculation)
		tuple<size_t, size_t> mcoords = mesh.get_mesh_coords_in_area(
			x[ind], z[ind],
			{ meshx == 0 ? 0 : meshx-1, meshz == 0 ? 0 : meshz-1 },
			{ min(mesh.nx-1, meshx+1), min(mesh.nz-1, meshz+1) }
		);
		meshx = get<0>(mcoords);
		meshz = get<1>(mcoords);

		// 3. stop the ray if it escapes the mesh
		if (x[ind] < mesh.xmin || x[ind] > mesh.xmax || z[ind] < mesh.zmin || z[ind] > mesh.zmax) break;

		ind++;
	}
	return ind;
}

double get_k(Mesh& mesh, double x0) {
	double wpe_interp = sqrt(
		interp_closure(
			[&] (size_t i) -> double { return mesh.points[i*mesh.nz].eden; },
			[&] (size_t i) -> double { return mesh.points[i*mesh.nz].x; }, mesh.nx,
			x0
		) * 1e6 * pow(consts::EC, 2) / (consts::ME*consts::E0)
	);

	return sqrt((pow(consts::OMEGA, 2) - pow(wpe_interp, 2)) / (pow(consts::C_SPEED, 2)));
}

/// May be able to be made nicer or eliminated entirely.
/// Assumes x either monotonically increases or decreases. y, x is a set of points.
/// Returns the interpolated y value for the x value given
///
/// NOTE: does not handle errors. panics instead.
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

// implementation of interp that takes closures, for get_k
template<typename Func1, typename Func2>
double interp_closure(Func1 y, Func2 x, size_t len, double xp) {
	if (x(0) <= x(len-1)) {
		// x monotonically increase
		if (xp <= x(0)) {
			return y(0);
		} else if (xp >= x(len-1)) {
			return y(len-1);
		}

		size_t low = 0;
		size_t high = len - 1;
		size_t mid = (low + high) >> 1;
		while (low < high - 1) {
			if (x(mid) >= xp) {
				high = mid;
			} else {
				low = mid;
			}
			mid = (low + high) >> 1;
		}

		assert((xp >= x(mid)) && (xp <= x(mid+1)));
		return y(mid) + ((y(mid+1) - y(mid)) / (x(mid+1) - x(mid)) * (xp - x(mid)));
	} else {
		if (xp >= x(0)) {
			return y(0);
		} else if (xp <= x(len-1)) {
			return y(len-1);
		}

		size_t low = 0;
		size_t high = len - 1;
		size_t mid = (low + high) >> 1;
		while (low < high - 1) {
			if (x(mid) <= xp) {
				low = mid;
			} else {
				high = mid;
			}
			mid = (low + high) >> 1;
		}

		assert((xp <= x(mid)) && (xp >= x(mid+1)));
		return y(mid) + ((y(mid+1)-y(mid))/(x(mid+1)-x(mid)) * (xp-x(mid)));
	}
}
