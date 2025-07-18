#include <cstddef>
#include <iostream>
#include <cassert>
#include <chrono>
#include <vector>
#include <numeric>
#include <omp.h>
#include "ray_trace.cuh"
#include "ray_trace.hpp"
#include "consts.hpp"
#include "structs.hpp"
#include "utils.cuh"
#include "omega_beams.cuh"

#define CEIL_DIV(a, b) ((a+b-1)/b)
#define THREADS_PER_BLOCK 256

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false) {
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

using namespace std::literals; // for dividing times by 1.0s

void ray_trace(MeshPoint* mesh, Crossing* crossings, size_t* turn, RaystorePt* raystore) {
	auto start_time = std::chrono::high_resolution_clock::now();
	// calculate dedens (derivative of eden wrt. x, y z)
	Xyz<double>* deden = new Xyz<double>[consts::GRID];
	for (size_t x = 1; x < consts::NX - 1; x++) {
		for (size_t y = 1; y < consts::NY - 1; y++) {
			for (size_t z = 1; z < consts::NZ - 1; z++) {
				Xyz<double>* pt = get_pt(deden, x, y, z);
				MeshPoint* xplusone = get_pt(mesh, x+1, y, z);
				MeshPoint* xminusone = get_pt(mesh, x-1, y, z);
				MeshPoint* yplusone = get_pt(mesh, x, y+1, z);
				MeshPoint* yminusone = get_pt(mesh, x, y-1, z);
				MeshPoint* zplusone = get_pt(mesh, x, y, z+1);
				MeshPoint* zminusone = get_pt(mesh, x, y, z-1);
				pt->x = (xplusone->eden - xminusone->eden)/(xplusone->pt.x - xminusone->pt.x);
				pt->y = (yplusone->eden - yminusone->eden)/(yplusone->pt.y - yminusone->pt.y);
				pt->z = (zplusone->eden - zminusone->eden)/(zplusone->pt.z - zminusone->pt.z);
			}
		}
	}
	// assumes cube
	for (size_t i = 0; i < consts::NX; i++) {
		for (size_t j = 0; j < consts::NY; j++) {
			get_pt(deden, 0, i, j)->x = get_pt(deden, 1, i, j)->x;
			get_pt(deden, i, 0, j)->y = get_pt(deden, i, 1, j)->y;
			get_pt(deden, i, j, 0)->z = get_pt(deden, i, j, 1)->z;
			get_pt(deden, consts::NX-1, i, j)->x = get_pt(deden, consts::NX-2, i, j)->x;
			get_pt(deden, i, consts::NX-1, j)->y = get_pt(deden, i, consts::NX-2, j)->y;
			get_pt(deden, i, j, consts::NX-1)->z = get_pt(deden, i, j, consts::NX-2)->z;
		}
	}

	// c++ has something about phase_r and pow_r that are not used

	// Create reference start positions to be rotated into place
	Xyz<double>* ref_positions = new Xyz<double>[consts::NRAYS];

	double dx = (consts::BEAM_MAX_Z - consts::BEAM_MIN_Z)/(consts::NRAYS_X - 1);
	double dy = (consts::BEAM_MAX_Z - consts::BEAM_MIN_Z)/(consts::NRAYS_Y - 1);
	for (size_t i = 0; i < consts::NRAYS_X; i++) {
		size_t ind = i * consts::NRAYS_Y;
		for (size_t j = 0; j < consts::NRAYS_Y; j++) {
			ref_positions[ind + j].x = consts::BEAM_MIN_Z + dx * j + (consts::DX/2);
			ref_positions[ind + j].y = consts::BEAM_MIN_Z + dy * i + (consts::DY/2);
			ref_positions[ind + j].z = consts::FOCAL_LENGTH - (consts::DZ/2);
		}
	}

	// See how much memory we need
	// TODO: update to enumerate GPUs, find the one with more memory? or less
	// memory? or, maybe, have the user provide a GPU profile to use, so the
	// code doesn't have to guess how they want to use their GPUs...?
	size_t real_gpu_bytes_free;
	gpuErrchk(cudaMemGetInfo(&real_gpu_bytes_free, NULL));
	// take out mesh size + ref positions and stuff
	// TODO: update when removing TEMPoffsets!
	size_t gpu_bytes_free = real_gpu_bytes_free -
		((sizeof(MeshPoint) + sizeof(Xyz<double>)) * consts::GRID +
		(sizeof(Xyz<double>) + sizeof(double)) * consts::NRAYS);
	size_t n_beams_in_memory = gpu_bytes_free / ((sizeof(Crossing) * consts::NCROSSINGS + sizeof(size_t)) * consts::NRAYS + sizeof(RaystorePt) * consts::GRID);
	size_t n_batches = CEIL_DIV(consts::NBEAMS, n_beams_in_memory);

	// make sure to use all available GPUs
	int device_count;
	gpuErrchk(cudaGetDeviceCount(&device_count));

	// enumerate devices, hacky way of making sure
	// that the one non-A100 GPU on darkworld isn't used....?
	std::vector<int> device_ids(device_count);
	std::iota(device_ids.begin(), device_ids.end(), 0);
	for (size_t i = 0; i < device_ids.size(); i++) {
		cudaDeviceProp device_prop;
		cudaGetDeviceProperties(&device_prop, device_ids[i]);
		if (device_prop.totalGlobalMem < gpu_bytes_free) {
			device_ids.erase(device_ids.begin()+i);
			i--;
		}
	}
	//device_count = device_ids.size(); TODO
	device_count = 1;
	n_batches = max((int)n_batches, device_count);
	n_beams_in_memory = CEIL_DIV(consts::NBEAMS, n_batches);

	printf("\t%d GPU(s)\n", device_count);
	printf("\t%lu batch(es), %lu beams per batch\n", n_batches, n_beams_in_memory);

	// NOTE: TEMPORARY: intensity offsets
	double* TEMPoffsets = new double[consts::NRAYS_X];
	double TEMPdx = (consts::BEAM_MAX_Z-consts::BEAM_MIN_Z)/(consts::NRAYS_X-1);
	for (size_t i = 0; i < consts::NRAYS_X; i++) {
		TEMPoffsets[i] = consts::BEAM_MIN_Z + i * TEMPdx;
	}

	omp_set_num_threads(device_count);

	#pragma omp parallel
	{
		int device = omp_get_thread_num();
		int device_id = device_ids[device];
		gpuErrchk(cudaSetDevice(device_id));

		MeshPoint* cuda_mesh;
		Crossing* cuda_crossings;
		Xyz<double>* cuda_deden;
		Xyz<double>* cuda_ref_positions;
		size_t* cuda_turn;
		Xyz<double>* cuda_child;
		double* cuda_dist;
		double* TEMPcuda_offsets;
		RaystorePt* cuda_raystore;

		gpuErrchk(cudaMalloc(&cuda_mesh, sizeof(MeshPoint) * consts::GRID));
		gpuErrchk(cudaMemcpy(cuda_mesh, mesh, sizeof(MeshPoint) * consts::GRID, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMalloc(&cuda_deden, sizeof(Xyz<double>) * consts::GRID));
		gpuErrchk(cudaMemcpy(cuda_deden, deden, sizeof(Xyz<double>) * consts::GRID, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMalloc(&cuda_ref_positions, sizeof(Xyz<double>) * consts::NRAYS));
		gpuErrchk(cudaMemcpy(cuda_ref_positions, ref_positions, sizeof(Xyz<double>) * consts::NRAYS, cudaMemcpyHostToDevice));

		gpuErrchk(cudaMalloc(&cuda_crossings, sizeof(Crossing) * consts::NCROSSINGS * consts::NRAYS * n_beams_in_memory));
		gpuErrchk(cudaMemset(cuda_crossings, 0, sizeof(Crossing) * consts::NCROSSINGS * consts::NRAYS * n_beams_in_memory));
		gpuErrchk(cudaMalloc(&cuda_turn, sizeof(size_t) * consts::NRAYS * n_beams_in_memory));
		gpuErrchk(cudaMemset(cuda_turn, 0, sizeof(size_t) * consts::NRAYS * n_beams_in_memory));

		gpuErrchk(cudaMalloc(&cuda_raystore, sizeof(RaystorePt) * consts::GRID * n_beams_in_memory));
		gpuErrchk(cudaMemset(cuda_raystore, 0, sizeof(RaystorePt) * consts::GRID * n_beams_in_memory));

		gpuErrchk(cudaMalloc(&TEMPcuda_offsets, sizeof(double) * consts::NRAYS_X));
		gpuErrchk(cudaMemcpy(TEMPcuda_offsets, TEMPoffsets, sizeof(double) * consts::NRAYS_X, cudaMemcpyHostToDevice));

		// fill the rest of the memory with trajectories
		size_t gpu_bytes_free; // local version
		gpuErrchk(cudaMemGetInfo(&gpu_bytes_free, NULL));
		size_t n_trajectories = gpu_bytes_free / ((sizeof(Xyz<double>) + sizeof(double)) * 2 * consts::NT);
		if (n_trajectories > consts::NRAYS) n_trajectories = consts::NRAYS;
		size_t rays_per_thread = CEIL_DIV(consts::NRAYS, n_trajectories);
		n_trajectories = CEIL_DIV(consts::NRAYS, rays_per_thread);
		// leave 1mb on gpu (i just chose a random number that sounds right)
		// (if things break make it bigger I guess)
		if (gpu_bytes_free - n_trajectories * (sizeof(Xyz<double>) + sizeof(double)) * 2 * consts::NT < 1000000) {
			n_trajectories--;
			rays_per_thread = CEIL_DIV(consts::NRAYS, n_trajectories);
			n_trajectories = CEIL_DIV(consts::NRAYS, rays_per_thread);
		}
		printf("\t%lu ray(s) per thread on GPU %d\n", rays_per_thread, device);

		gpuErrchk(cudaMalloc(&cuda_child, sizeof(Xyz<double>) * consts::NT * 2 * n_trajectories));
		gpuErrchk(cudaMalloc(&cuda_dist, sizeof(double) * consts::NT * 2 * n_trajectories));

		for (size_t batch = 0; batch < n_batches; batch += device_count) {
			if (batch + device >= n_batches) break;
			for (size_t beamnum = 0; beamnum < n_beams_in_memory; beamnum++) {
				//if (beamnum + (batch + device) * n_beams_in_memory != 0) continue;
				trace_rays
					<<<CEIL_DIV(consts::NRAYS / rays_per_thread, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>
					(cuda_mesh, cuda_deden, cuda_child, cuda_dist,
					 beamnum, beamnum + (batch + device) * n_beams_in_memory,
					 cuda_ref_positions, TEMPcuda_offsets, cuda_crossings, cuda_turn, rays_per_thread,
					 cuda_raystore);
				gpuErrchk(cudaPeekAtLastError());
			}
			gpuErrchk(cudaDeviceSynchronize());
			gpuErrchk(cudaMemcpy(crossings + (batch + device) * n_beams_in_memory * consts::NCROSSINGS * consts::NRAYS, cuda_crossings,
				sizeof(Crossing) * consts::NCROSSINGS * consts::NRAYS * n_beams_in_memory, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(cuda_crossings, 0, sizeof(Crossing) * consts::NCROSSINGS * consts::NRAYS * n_beams_in_memory));
			gpuErrchk(cudaMemcpy(turn + (batch + device) * n_beams_in_memory * consts::NRAYS, cuda_turn,
				sizeof(size_t) * consts::NRAYS * n_beams_in_memory, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(raystore + (batch + device) * n_beams_in_memory * consts::GRID, cuda_raystore,
				sizeof(RaystorePt) * consts::GRID * n_beams_in_memory, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(cuda_raystore, 0, sizeof(RaystorePt) * consts::GRID * n_beams_in_memory));
		}

		gpuErrchk(cudaFree(cuda_mesh));
		gpuErrchk(cudaFree(cuda_crossings));
		gpuErrchk(cudaFree(cuda_deden));
		gpuErrchk(cudaFree(cuda_ref_positions));
		gpuErrchk(cudaFree(cuda_turn));
		gpuErrchk(cudaFree(cuda_child));
		gpuErrchk(cudaFree(cuda_dist));
		gpuErrchk(cudaFree(cuda_raystore));
		gpuErrchk(cudaFree(TEMPcuda_offsets));
	}

	delete[] TEMPoffsets;
	delete[] deden;
	delete[] ref_positions;

	auto end_time = std::chrono::high_resolution_clock::now();
	printf("\tTotal time: %Lf seconds\n", (end_time - start_time) / 1.0s);
}

__global__ void trace_rays(MeshPoint* mesh, Xyz<double>* deden,
		Xyz<double>* child, double* dist,
		size_t b_mem_num, size_t beamnum, Xyz<double>* ref_positions, double* TEMPoffsets,
		Crossing* crossings, size_t* turn, size_t rays_per_thread,
		RaystorePt* raystore) {
	size_t thread_num = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;
	// is this okay? will the compiler optimize this? let's hope...
	size_t max_raynum = min(consts::NRAYS, (thread_num+1) * rays_per_thread);
	/*Xyz<double> k0 = {
		-1 * BEAM_NORM[beamnum][0],
		-1 * BEAM_NORM[beamnum][1],
		-1 * BEAM_NORM[beamnum][2]
	};*/
	Xyz<double> k0 = {
		consts::KX01[beamnum], consts::KY01[beamnum], consts::KZ01[beamnum]
	};
	Xyz<double>* child1 = child + (consts::NT * thread_num * 2);
	Xyz<double>* child2 = child1 + consts::NT;
	double* dist1 = dist + (consts::NT * thread_num * 2);
	double* dist2 = dist1 + consts::NT;
	for (size_t raynum = thread_num * rays_per_thread; raynum < max_raynum; raynum++) {
	/*
		// get ray start pos
		Xyz<double>* ref_pos = &(ref_positions[raynum]);
		Xyz<double> start_pos = *ref_pos;
		// ensure circular beam with power cutoff
		// TODO: FIGURE OUT SOMETHING BETTER THAN THIS
		// BECAUSE OF CUDA!!!!!!!
		double ref = sqrt(pow(start_pos.x, 2) + pow(start_pos.y, 2));
		if (ref >= consts::BEAM_MAX_Z) continue;

		// C++ comment reads: "We will first rotate in XZ plane about the Y axis,
		// so z -> z' will be complete. Then, we will rotate in the XY plane about
		// the Z axis so x -> x' and y -> y' will be complete."
		// first rotation
		double theta1 = acos(BEAM_NORM[beamnum][2]);
		start_pos.x = ref_pos->x*cos(theta1) + ref_pos->z*sin(theta1);
		start_pos.z = ref_pos->z*cos(theta1) - ref_pos->x*sin(theta1);
		// second rotation
		double theta2 = atan2(BEAM_NORM[beamnum][1] * consts::FOCAL_LENGTH,
			BEAM_NORM[beamnum][0] * consts::FOCAL_LENGTH);
		double tmp_x0 = start_pos.x;
		start_pos.x = start_pos.x*cos(theta2) - start_pos.y*sin(theta2);
		start_pos.y = start_pos.y*cos(theta2) + tmp_x0*sin(theta2);*/

		double dx = (consts::BEAM_MAX_Z - consts::BEAM_MIN_Z)/(consts::NRAYS_X - 1);
		Xyz<double> start_pos = {
			consts::X01[beamnum] + consts::BEAM_MIN_Z + dx * (raynum / consts::NRAYS_X),
			consts::Y01[beamnum],
			consts::Z01[beamnum] + consts::BEAM_MIN_Z + dx * (raynum % consts::NRAYS_X)
		};
		if (beamnum == 1) {
			start_pos = {start_pos.y, start_pos.x, start_pos.z};
		}

		// launch child rays
		constexpr double dist = 0.2 * consts::DX;
		Xyz<double> delta1 = {-k0.y, k0.x, 0};
		double l1 = dist / mag(delta1);
		delta1.x *= l1;
		delta1.y *= l1;
		Xyz<double> child_start_pos = {
			start_pos.x + delta1.x,
			start_pos.y + delta1.y,
			start_pos.z + delta1.z
		};

		size_t c1_size = launch_child_ray(mesh, deden,
			child_start_pos, k0,
			child1, dist1);
		delta1 = {-k0.x * k0.z, -k0.y * k0.z, k0.x * k0.x + k0.y * k0.y};
		l1 = dist / mag(delta1);
		delta1.x *= l1;
		delta1.y *= l1;
		delta1.z *= l1;
		child_start_pos = {
			start_pos.x + delta1.x,
			start_pos.y + delta1.y,
			start_pos.z + delta1.z
		};
		size_t c2_size = launch_child_ray(mesh, deden,
			child_start_pos, k0,
			child2, dist2);

		// launch parent ray
		Crossing* crossings_ind = crossings + (b_mem_num*consts::NRAYS + raynum)*consts::NCROSSINGS;
		// NOTE: TEMPORARY: compute initial intensity
		double TEMPoffset = sqrt(pow(TEMPoffsets[raynum / consts::NRAYS_X], 2) + pow(TEMPoffsets[raynum % consts::NRAYS_X], 2));
		double TEMPinit_intensity = (consts::INTENSITY/1e14)*exp(-2*pow(abs(TEMPoffset/consts::SIGMA),4.0));
		size_t TEMPcnum =
		launch_parent_ray(mesh, deden,
			child1, dist1, c1_size,
			child2, dist2, c2_size,
			start_pos, k0, crossings_ind, turn + b_mem_num*consts::NRAYS + raynum, TEMPinit_intensity,
			raystore + b_mem_num*consts::GRID, raynum);
	}
}

__device__ size_t launch_parent_ray(MeshPoint* mesh, Xyz<double>* deden,
		Xyz<double>* child1, double* dist1, size_t c1_size,
		Xyz<double>* child2, double* dist2, size_t c2_size,
		Xyz<double> pos, Xyz<double> k0, Crossing* crossings, size_t* turn, double TEMPintensity,
		RaystorePt* raystore, size_t raynum) {
	size_t cnum = 0;
	double max_turn_amp = 0.0;
	size_t i_max_turn_amp = 0; // crossing index of max turn amp?

	// pos = start_pos already (by value, not reference)
	
	// compute init_area "getTriangleArea" in Shuang's code
	Xyz<double> ab = {
		child1[0].x - pos.x, child1[0].y - pos.y, child1[0].z - pos.z
	};
	Xyz<double> ac = {
		child2[0].x - pos.x, child2[0].y - pos.y, child2[0].z - pos.z
	};
	double init_area = 0.5 * sqrt(pow(ab.y * ac.z - ab.z * ac.y, 2) +
			pow(ab.z * ac.x - ab.x * ac.z, 2) +
			pow(ab.x * ac.y - ab.y * ac.x, 2));

	Xyz<size_t> mesh_pos = get_mesh_coords(mesh, pos);
	MeshPoint* mesh_pt = get_pt(mesh, mesh_pos);

	double k = get_k(mesh, pos, mesh_pos);
	double knorm = mag(k0);
	Xyz<double> vel = {
		pow(consts::C_SPEED, 2) * ((k0.x / knorm) * k) / consts::OMEGA,
		pow(consts::C_SPEED, 2) * ((k0.y / knorm) * k) / consts::OMEGA,
		pow(consts::C_SPEED, 2) * ((k0.z / knorm) * k) / consts::OMEGA,
	};

	double curr_dist = 0.0;
	double kds = 0.0;
	double next_kds = 0.0;
	//double phase = 0.0;
	double permittivity = 1 - mesh_pt->eden / consts::NCRIT;
	if (permittivity < 0) permittivity = 0;

	for (size_t _ = 1; _ < consts::NT; _++) {
		// Update velocity and position
		Xyz<double>my_deden = {
			interp3D(
				pos, mesh_pos,
				[&] (size_t x, size_t y, size_t z) -> double {
					return get_pt(deden, x, y, z)->x;
				}
			),
			interp3D(
				pos, mesh_pos,
				[&] (size_t x, size_t y, size_t z) -> double {
					return get_pt(deden, x, y, z)->y;
				}
			),
			interp3D(
				pos, mesh_pos,
				[&] (size_t x, size_t y, size_t z) -> double {
					return get_pt(deden, x, y, z)->z;
				}
			)
		};
		Xyz<double> prev_vel = vel;
		vel = {
			vel.x - pow(consts::C_SPEED, 2) / (2.0 * consts::NCRIT) * my_deden.x * consts::DT,
			vel.y - pow(consts::C_SPEED, 2) / (2.0 * consts::NCRIT) * my_deden.y * consts::DT,
			vel.z - pow(consts::C_SPEED, 2) / (2.0 * consts::NCRIT) * my_deden.z * consts::DT
		};
		Xyz<double> prev_pos = pos;
		pos = {
			pos.x + vel.x * consts::DT,
			pos.y + vel.y * consts::DT,
			pos.z + vel.z * consts::DT
		};
		if (pos.x < consts::XMIN || pos.x > consts::XMAX ||
				pos.y < consts::YMIN || pos.y > consts::YMAX ||
				pos.z < consts::ZMIN || pos.z > consts::ZMAX) {
			*turn = i_max_turn_amp;
			break;
		}
		
		// Update mesh position
		Xyz<size_t> prev_mesh_pos = mesh_pos;
		mesh_pos = get_mesh_coords(mesh, pos);
		mesh_pt = get_pt(mesh, mesh_pos);

		double prev_permittivity = permittivity;
		permittivity = 1 - mesh_pt->eden / consts::NCRIT;
		if (permittivity < 0.0) permittivity = 0.0;
		double distance_travelled = sqrt(pow(pos.x - prev_pos.x, 2) +
				pow(pos.y - prev_pos.y, 2) +
				pow(pos.z - prev_pos.z, 2));
		// TODO: interp3d?
		next_kds -= distance_travelled * 1e4 *
			interp3D(pos, mesh_pos, [&] (size_t x, size_t y, size_t z) -> double {
				return get_pt(mesh, x, y, z)->kib;
			}) /
			sqrt(interp3D(pos, mesh_pos, [&] (size_t x, size_t y, size_t z) -> double {
				return get_pt(mesh, x, y, z)->dielectric;
			}));
		//next_kds -= distance_travelled * get_pt(mesh, mesh_pos)->kib_multiplier;

		double c_perms[3];
		double c_distances[3]; // to sort by
		Crossing crosses[3];
		size_t this_cnum = 0;

		// 3 maximum crosses, then sorted at the end by their... boxes? x then y then z...?
		// closure that modifies the crossing at index cnum,
		// and then increments cnum.
		auto new_crossing = [&] (Xyz<double> c_pos, Xyz<size_t> c_mesh_pos, double frac) {
			// first, get mesh pos of crossing (cross[xyz]Ind)
			crosses[this_cnum].pt = c_pos;
			crosses[this_cnum].boxes = c_mesh_pos;
			// then, compute localAreaRatio, localKds, localEnergy, localPermittivity
			double distance_to_crossing = sqrt(pow(c_pos.x - prev_pos.x, 2) +
					pow(c_pos.y - prev_pos.y, 2) + pow(c_pos.z - prev_pos.z, 2));
			c_distances[this_cnum] = distance_to_crossing;
			Xyz<double> childp1 = interp_xyz(child1, dist1, c1_size,
					curr_dist + distance_to_crossing);
			Xyz<double> childp2 = interp_xyz(child2, dist2, c2_size,
					curr_dist + distance_to_crossing);

			// localPermittivity
			c_perms[this_cnum] = frac * permittivity + (1-frac) * prev_permittivity;

			// "getProjectedTriangleArea" which takes c_pos, childp1/2, interpk
			Xyz<double> interpk = {
				frac * vel.x + (1-frac) * prev_vel.x,
				frac * vel.y + (1-frac) * prev_vel.y,
				frac * vel.z + (1-frac) * prev_vel.z
			};
			double kmag = mag(interpk);
			ab = {childp1.x - c_pos.x, childp1.y - c_pos.y, childp1.z - c_pos.z};
			ac = {childp2.x - c_pos.x, childp2.y - c_pos.y, childp2.z - c_pos.z};
			Xyz<double> tri_norm = { // called just [xyz]norm in Shuang's code
				ab.y * ac.z - ab.z * ac.y,
				ab.z * ac.x - ab.x * ac.z,
				ab.x * ac.y - ab.y * ac.x
			};
			if ((1 - c_perms[this_cnum]) < 0.8) {
				crosses[this_cnum].area_ratio = (0.5 * abs(
						tri_norm.x * interpk.x +
						tri_norm.y * interpk.y +
						tri_norm.z * interpk.z) / kmag)
					/ init_area;
			} else {
				crosses[this_cnum].area_ratio = 0.5 * sqrt(pow(tri_norm.x, 2) + pow(tri_norm.y, 2) + pow(tri_norm.z, 2)) / init_area;
			}
			crosses[this_cnum].kds = exp(frac * next_kds + (1-frac) * kds);
			crosses[this_cnum].i_b = TEMPintensity * crosses[this_cnum].kds;

			this_cnum++;
		};

		if ((pos.x > mesh_pt->pt.x && prev_pos.x <= mesh_pt->pt.x) ||
				(pos.x < mesh_pt->pt.x && prev_pos.x >= mesh_pt->pt.x) ||
				(mesh_pos.x != prev_mesh_pos.x)) {
			double crossx = mesh_pt->pt.x;
			if (!((pos.x > crossx && prev_pos.x <= crossx) ||
						(pos.x < crossx && prev_pos.x >= crossx))) {
				crossx = get_pt(mesh, prev_mesh_pos)->pt.x;
			}
			Xyz<double> c_pos = {
				crossx,
				prev_pos.y + // crossy
					((pos.y - prev_pos.y) / (pos.x - prev_pos.x) * (crossx - prev_pos.x)),
				prev_pos.z + // crossz
					((pos.z - prev_pos.z) / (pos.x - prev_pos.x) * (crossx - prev_pos.x))
			};
			Xyz<size_t> c_mesh_pos = get_mesh_coords(mesh, c_pos);
			c_mesh_pos.x = mesh_pos.x;
			new_crossing(
				c_pos, c_mesh_pos,
				(crossx - prev_pos.x) / (pos.x - prev_pos.x) // frac
			);
		}
		if ((pos.y > mesh_pt->pt.y && prev_pos.y <= mesh_pt->pt.y) ||
				(pos.y < mesh_pt->pt.y && prev_pos.y >= mesh_pt->pt.y) ||
				(mesh_pos.y != prev_mesh_pos.y)) {
			double crossy = mesh_pt->pt.y;
			if (!((pos.y > crossy && prev_pos.y <= crossy) ||
						(pos.y < crossy && prev_pos.y >= crossy))) {
				crossy = get_pt(mesh, prev_mesh_pos)->pt.y;
			}
			Xyz<double> c_pos = {
				prev_pos.x + // crossx
					((pos.x - prev_pos.x) / (pos.y - prev_pos.y) * (crossy - prev_pos.y)),
				crossy,
				prev_pos.z + // crossz
					((pos.z - prev_pos.z) / (pos.y - prev_pos.y) * (crossy - prev_pos.y))
			};
			Xyz<size_t> c_mesh_pos = get_mesh_coords(mesh, c_pos);
			c_mesh_pos.y = mesh_pos.y;
			new_crossing(
				c_pos, c_mesh_pos,
				(crossy - prev_pos.y) / (pos.y - prev_pos.y) // frac
			);
		}
		if ((pos.z > mesh_pt->pt.z && prev_pos.z <= mesh_pt->pt.z) ||
				(pos.z < mesh_pt->pt.z && prev_pos.z >= mesh_pt->pt.z) ||
				(mesh_pos.z != prev_mesh_pos.z)) {
			double crossz = mesh_pt->pt.z;
			if (!((pos.z > crossz && prev_pos.z <= crossz) ||
						(pos.z < crossz && prev_pos.z >= crossz))) {
				crossz = get_pt(mesh, prev_mesh_pos)->pt.z;
			}
			Xyz<double> c_pos = {
				prev_pos.x +
					((pos.x - prev_pos.x) / (pos.z - prev_pos.z) * (crossz - prev_pos.z)),
				prev_pos.y +
					((pos.y - prev_pos.y) / (pos.z - prev_pos.z) * (crossz - prev_pos.z)),
				crossz
			};
			Xyz<size_t> c_mesh_pos = get_mesh_coords(mesh, c_pos);
			c_mesh_pos.z = mesh_pos.z;
			new_crossing(
				c_pos, c_mesh_pos,
				(crossz - prev_pos.z) / (pos.z - prev_pos.z)
			);
		}

		// sort crosses
		// update maxturnamp based on arearatio and permittivity
		if (this_cnum == 3) {
			// https://stackoverflow.com/questions/4793251/sorting-int-array-with-only-3-elements
			// I used the second answer, which is more optimal... and ugly... SORRY!!!!
			Crossing temp;
			double temp_perm;
			double temp_dist;
			if (c_distances[0] < c_distances[1]) {
				if (c_distances[1] > c_distances[2]) {
					if (c_distances[0] < c_distances[2]) {
						swap(crosses[1], crosses[2]);
						swap(c_perms[1], c_perms[2]);
						swap(c_distances[1], c_distances[2]);
					} else {
						temp = crosses[0]; temp_perm = c_perms[0]; temp_dist = c_distances[0];
						crosses[0] = crosses[2]; c_perms[0] = c_perms[2]; c_distances[0] = c_distances[2];
						crosses[2] = crosses[1]; c_perms[2] = c_perms[1]; c_distances[2] = c_distances[1];
						crosses[1] = temp; c_perms[1] = temp_perm; c_distances[1] = temp_dist;
					}
				}
			} else {
				if (c_distances[1] < c_distances[2]) {
					if (c_distances[0] < c_distances[2]) {
						swap(crosses[0], crosses[1]);
						swap(c_perms[0], c_perms[1]);
						swap(c_distances[0], c_distances[1]);
					} else {
						temp = crosses[0]; temp_perm = c_perms[0]; temp_dist = c_distances[0];
						crosses[0] = crosses[1]; c_perms[0] = c_perms[1]; c_distances[0] = c_distances[1];
						crosses[1] = crosses[2]; c_perms[1] = c_perms[2]; c_distances[1] = c_distances[2];
						crosses[2] = temp; c_perms[2] = temp_perm; c_distances[2] = temp_dist;
					}
				} else {
					swap(crosses[0], crosses[2]);
					swap(c_perms[0], c_perms[2]);
					swap(c_distances[0], c_distances[2]);
				}
			}
		} else if (this_cnum == 2) {
			if (c_distances[0] > c_distances[1]) {
				swap(crosses[0], crosses[1]);
				swap(c_perms[1], c_perms[0]);
				swap(c_distances[1], c_distances[0]);
			}
		}
		for (size_t i = 0; i < this_cnum; i++) {
			// calc_dk
			if (cnum > 0) {
				Xyz<double> dk = {
					crosses[i].pt.x - crossings[cnum-1].pt.x,
					crosses[i].pt.y - crossings[cnum-1].pt.y,
					crosses[i].pt.z - crossings[cnum-1].pt.z
				};
				double dkmag = mag(dk);
				if (dkmag == 0) continue;
				crossings[cnum-1].dk = {dk.x/dkmag, dk.y/dkmag, dk.z/dkmag};
				crossings[cnum-1].dkmag = dkmag * 10000.0;
			}
			crossings[cnum] = crosses[i];
			if (crossings[cnum].boxes.x == mesh_pos.x &&
					crossings[cnum].boxes.y == mesh_pos.y &&
					crossings[cnum].boxes.z == mesh_pos.z) {
				RaystorePt* rspt = get_pt(raystore, crossings[cnum].boxes);
				if (rspt->raynum == 0 || raynum < rspt->raynum) { // TEMP WILLOW
					rspt->raynum = raynum;
					rspt->cnum = cnum;
				}
			}
			double ray_amp = 1 / (crossings[cnum].area_ratio * c_perms[i]);
			if (ray_amp > max_turn_amp) {
				max_turn_amp = ray_amp;
				i_max_turn_amp = cnum-i;
			}
			cnum++;
		}
		// update currDist, kds, totalEnergy based on step_size = distance travelled.
		curr_dist += distance_travelled;
		kds = next_kds;
	}
	return cnum;
}

// returns the length of the trajectory reached
__device__ size_t launch_child_ray(MeshPoint* mesh, Xyz<double>* deden,
		Xyz<double> start_pos, Xyz<double> k0,
		Xyz<double>* traj, double* dist) {
	size_t ind = 0;

	dist[ind] = 0.0;
	traj[0] = start_pos;
	ind++;
	
	Xyz<size_t> mesh_pos = get_mesh_coords(mesh, traj[0]);

	double k = get_k(mesh, traj[0], mesh_pos);
	double knorm = mag(k0);
	Xyz<double> vel = {
		pow(consts::C_SPEED, 2) * ((k0.x / knorm) * k) / consts::OMEGA,
		pow(consts::C_SPEED, 2) * ((k0.y / knorm) * k) / consts::OMEGA,
		pow(consts::C_SPEED, 2) * ((k0.z / knorm) * k) / consts::OMEGA,
	};
	while (ind < consts::NT) {
		// 1. Calculate velocity and position at current timestamp
		// interpolating deden possibly can be made more efficient but
		// probably not the biggest math sink?
		// (redundancy is calculating the pos and mesh_pos update
		// which is not too much)
		Xyz<double>my_deden = {
			interp3D(
				traj[ind-1], mesh_pos,
				[&] (size_t x, size_t y, size_t z) -> double {
					return get_pt(deden, x, y, z)->x;
				}
			),
			interp3D(
				traj[ind-1], mesh_pos,
				[&] (size_t x, size_t y, size_t z) -> double {
					return get_pt(deden, x, y, z)->y;
				}
			),
			interp3D(
				traj[ind-1], mesh_pos,
				[&] (size_t x, size_t y, size_t z) -> double {
					return get_pt(deden, x, y, z)->z;
				}
			),
		};
		vel = {
			vel.x - pow(consts::C_SPEED, 2) / (2.0 * consts::NCRIT) * my_deden.x * consts::DT,
			vel.y - pow(consts::C_SPEED, 2) / (2.0 * consts::NCRIT) * my_deden.y * consts::DT,
			vel.z - pow(consts::C_SPEED, 2) / (2.0 * consts::NCRIT) * my_deden.z * consts::DT
		};
		traj[ind] = {
			traj[ind-1].x + vel.x * consts::DT,
			traj[ind-1].y + vel.y * consts::DT,
			traj[ind-1].z + vel.z * consts::DT
		};
		dist[ind] = dist[ind-1] + sqrt(
			pow(traj[ind].x - traj[ind-1].x, 2) +
			pow(traj[ind].y - traj[ind-1].y, 2) +
			pow(traj[ind].z - traj[ind-1].z, 2)
		);

		// 2. update meshx, meshz (for vel calculation)
		// This used to be get mesh coords in area when search was being done
		// but the new system (assumes a linearly spaced mesh) is quick w/o search
		mesh_pos = get_mesh_coords(mesh, traj[ind]);

		// 3. stop the ray if it escapes the mesh
		if (traj[ind].x < consts::XMIN || traj[ind].x > consts::XMAX ||
			traj[ind].y < consts::YMIN || traj[ind].y > consts::YMAX ||
			traj[ind].z < consts::ZMIN || traj[ind].z > consts::ZMAX) break;

		ind++;
	}
	return ind;
}

__device__ double get_k(MeshPoint* mesh, Xyz<double> pos, Xyz<size_t> mesh_pos) {
	double interpolated_eden = interp3D(
		pos, mesh_pos,
		[&] (size_t x, size_t y, size_t z) -> double {
			return get_pt(mesh, x, y, z)->eden;
		}
	);

	double wpe_interp = sqrt(interpolated_eden * 1e6 * pow(consts::EC, 2) / (consts::ME * consts::E0));
	return sqrt((pow(consts::OMEGA, 2) - pow(wpe_interp, 2)) / pow(consts::C_SPEED, 2));
}

// used for interpolating child distance
__device__ Xyz<double> interp_xyz(Xyz<double>* points, double* dist, size_t len, double xp) {
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
