#include <cstddef>
#include <iostream>
#include <cassert>
#include <chrono>
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
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

using namespace std::literals; // for dividing times by 1.0s

void ray_trace(MeshPoint* mesh, RayTraceResult* result) {
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
			ref_positions[ind + j].z = consts::FOCAL_LENGTH - (consts::DZ/2);
			ref_positions[ind + j].x = consts::BEAM_MIN_Z + dx * j + (consts::DX/2);
			ref_positions[ind + j].y = consts::BEAM_MIN_Z + dy * j + (consts::DY/2);
		}
	}

	auto time1 = std::chrono::high_resolution_clock::now();
	printf("\tSetting up arrays took %Lf seconds\n", (time1-start_time)/1.0s);

	// See how much memory we need
	size_t real_gpu_bytes_free;
	gpuErrchk(cudaMemGetInfo(&real_gpu_bytes_free, NULL));
	// take out mesh size + ref positions and stuff
	// TODO: update when removing TEMPoffsets!
	size_t gpu_bytes_free = real_gpu_bytes_free -
		((sizeof(MeshPoint) + sizeof(Xyz<double>)) * consts::GRID +
		(sizeof(Xyz<double>) + sizeof(double)) * consts::NRAYS);
	size_t n_beams_in_memory = gpu_bytes_free / ((sizeof(Crossing) * consts::NCROSSINGS + sizeof(size_t)) * consts::NRAYS);
	size_t n_batches = CEIL_DIV(consts::NBEAMS, n_beams_in_memory);
	n_beams_in_memory = CEIL_DIV(consts::NBEAMS, n_batches);

	int device_count = 1;
	if (n_batches > 1) {
		gpuErrchk(cudaGetDeviceCount(&device_count));
		device_count = min(device_count, (int)n_batches);
		printf("\tUsing %d GPUs\n", device_count);
		// adjust gpu memory free to fit smallest GPU
		// NOTE: this code assumes GPUs have roughly the same memory size/free memory
		// and this is just insurance in case one GPU is slightly less free
		// but if one GPU is significantly smaller, THIS WHOLE THING NEEDS TO BE CHANGED!!
		if (device_count > 1) {
			for (int device = 0; device < device_count; device++) {
				gpuErrchk(cudaSetDevice(device));
				size_t this_gpu_bytes_free;
				gpuErrchk(cudaMemGetInfo(&this_gpu_bytes_free, NULL));
				if (this_gpu_bytes_free < real_gpu_bytes_free) {
					gpu_bytes_free -= (real_gpu_bytes_free - this_gpu_bytes_free);
					real_gpu_bytes_free = this_gpu_bytes_free;
				}
			}
		}
	}
	printf("\tUsing %lu batch(es), %lu beams per batch\n", n_batches, n_beams_in_memory);

	MeshPoint** cuda_mesh = new MeshPoint*[device_count];
	Crossing** cuda_crossings = new Crossing*[device_count];
	Xyz<double>** cuda_deden = new Xyz<double>*[device_count];
	Xyz<double>** cuda_ref_positions = new Xyz<double>*[device_count];
	size_t** cuda_turn = new size_t*[device_count];

	// NOTE: TEMPORARY: intensity offsets
	double* TEMPoffsets = new double[consts::NRAYS];
	double TEMPdx = (consts::BEAM_MAX_Z-consts::BEAM_MIN_Z)/(consts::NRAYS_X-1);
	for (size_t i = 0; i < consts::NRAYS; i++) {
		TEMPoffsets[i] = consts::BEAM_MIN_Z + i * TEMPdx;
	}
	double** TEMPcuda_offsets = new double*[device_count];

	for (int device = 0; device < device_count; device++) {
		gpuErrchk(cudaSetDevice(device));

		gpuErrchk(cudaMalloc(cuda_mesh + device, sizeof(MeshPoint) * consts::GRID));
		gpuErrchk(cudaMemcpy(cuda_mesh[device], mesh, sizeof(MeshPoint) * consts::GRID, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMalloc(cuda_deden + device, sizeof(Xyz<double>) * consts::GRID));
		gpuErrchk(cudaMemcpy(cuda_deden[device], deden, sizeof(Xyz<double>) * consts::GRID, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMalloc(cuda_ref_positions + device, sizeof(Xyz<double>) * consts::NRAYS));
		gpuErrchk(cudaMemcpy(cuda_ref_positions[device], ref_positions, sizeof(Xyz<double>) * consts::NRAYS, cudaMemcpyHostToDevice));

		gpuErrchk(cudaMalloc(cuda_crossings + device, sizeof(Crossing) * consts::NCROSSINGS * consts::NRAYS * n_beams_in_memory));
		gpuErrchk(cudaMemset(cuda_crossings[device], 0, sizeof(Crossing) * consts::NCROSSINGS * consts::NRAYS * n_beams_in_memory));
		gpuErrchk(cudaMalloc(cuda_turn + device, sizeof(size_t) * consts::NRAYS * n_beams_in_memory));
		gpuErrchk(cudaMemset(cuda_turn[device], 0, sizeof(size_t) * consts::NRAYS * n_beams_in_memory));

		gpuErrchk(cudaMalloc(TEMPcuda_offsets + device, sizeof(double) * consts::NRAYS));
		gpuErrchk(cudaMemcpy(TEMPcuda_offsets[device], TEMPoffsets, sizeof(double) * consts::NRAYS, cudaMemcpyHostToDevice));

		gpuErrchk(cudaMemGetInfo(&real_gpu_bytes_free, NULL));
		gpu_bytes_free = min(gpu_bytes_free, real_gpu_bytes_free);
	}
	
	// fill the rest of the memory with child trajectories
	Xyz<double>** cuda_child = new Xyz<double>*[device_count];
	double** cuda_dist = new double*[device_count];

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
	printf("\tUsing %lu ray(s) per thread\n", rays_per_thread);

	for (int device = 0; device < device_count; device++) {
		gpuErrchk(cudaSetDevice(device));
		gpuErrchk(cudaMalloc(cuda_child + device, sizeof(Xyz<double>) * consts::NT * 2 * n_trajectories));
		gpuErrchk(cudaMalloc(cuda_dist + device, sizeof(double) * consts::NT * 2 * n_trajectories));
	}

	auto time2 = std::chrono::high_resolution_clock::now();
	printf("\tSetting up GPU memory and copying took %Lf seconds\n", (time2 - time1) / 1.0s);

	// allocate results
	Crossing** crossings = new Crossing*[n_batches];
	size_t** turn = new size_t*[n_batches];

	// NOTE: ideally: launch two threads, and copying can be done independently!
	// otherwise, right now, it assumes that ray tracing takes about the same amnt
	// of time on both GPUs.
	for (size_t batch = 0; batch < n_batches; batch += device_count) {
		for (int device = 0; device < device_count; device++) {
			if (batch + device >= n_batches) break;
			gpuErrchk(cudaSetDevice(device));
			for (size_t beamnum = 0; beamnum < n_beams_in_memory; beamnum++) {
				size_t real_beamnum = beamnum + (batch + device) * n_beams_in_memory;
				if (real_beamnum >= consts::NBEAMS) break;
				trace_rays
					<<<CEIL_DIV(consts::NRAYS / rays_per_thread, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>
					(cuda_mesh[device], cuda_deden[device], cuda_child[device], cuda_dist[device],
					 beamnum, real_beamnum,
					 cuda_ref_positions[device], TEMPcuda_offsets[device], cuda_crossings[device], cuda_turn[device], rays_per_thread);
				gpuErrchk(cudaPeekAtLastError());
			}
		}
		for (int device = 0; device < device_count; device++) {
			if (batch + device >= n_batches) break;
			gpuErrchk(cudaSetDevice(device));
			gpuErrchk(cudaDeviceSynchronize());

			// allocate results
			crossings[batch + device] = new Crossing[n_beams_in_memory * consts::NRAYS * consts::NCROSSINGS];
			turn[batch + device] = new size_t[n_beams_in_memory * consts::NRAYS];

			gpuErrchk(cudaMemcpy(crossings[batch + device], cuda_crossings[device],
				sizeof(Crossing) * consts::NCROSSINGS * consts::NRAYS * n_beams_in_memory, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(cuda_crossings[device], 0, sizeof(Crossing) * consts::NCROSSINGS * consts::NRAYS * n_beams_in_memory));
			gpuErrchk(cudaMemcpy(turn[batch + device], cuda_turn[device],
				sizeof(size_t) * consts::NRAYS * n_beams_in_memory, cudaMemcpyDeviceToHost));
			printf("\t\tDone batch %lu\n", batch + device + 1);
		}
	}

	gpuErrchk(cudaDeviceSynchronize());

	for (int device = 0; device < device_count; device++) {
		gpuErrchk(cudaSetDevice(device));
		gpuErrchk(cudaFree(cuda_mesh[device]));
		gpuErrchk(cudaFree(cuda_crossings[device]));
		gpuErrchk(cudaFree(cuda_deden[device]));
		gpuErrchk(cudaFree(cuda_ref_positions[device]));
		gpuErrchk(cudaFree(cuda_turn[device]));
		gpuErrchk(cudaFree(cuda_child[device]));
		gpuErrchk(cudaFree(cuda_dist[device]));
		gpuErrchk(cudaFree(TEMPcuda_offsets[device]));
	}

	delete[] TEMPoffsets;
	delete[] deden;
	delete[] ref_positions;

	delete[] cuda_mesh;
	delete[] cuda_crossings;
	delete[] cuda_deden;
	delete[] cuda_ref_positions;
	delete[] cuda_turn;
	delete[] cuda_child;
	delete[] cuda_dist;
	delete[] TEMPcuda_offsets;

	time1 = std::chrono::high_resolution_clock::now();
	printf("\tRay tracing + copying took %Lf seconds\n", (time1 - time2) / 1.0s);
	printf("\tTotal time: %Lf seconds\n", (time1 - start_time) / 1.0s);

	result->crossings = crossings;
	result->turn = turn;
	result->n_batches = n_batches;
	result->n_beams_per_batch = n_beams_in_memory;
}

void free_rt_result(RayTraceResult* result) {
	for (size_t i = 0; i < result->n_batches; i++) {
		delete[] result->crossings[i];
		delete[] result->turn[i];
	}
	delete[] result->crossings;
	delete[] result->turn;
}

__global__ void trace_rays(MeshPoint* mesh, Xyz<double>* deden,
		Xyz<double>* child, double* dist,
		size_t b_mem_num, size_t beamnum, Xyz<double>* ref_positions, double* TEMPoffsets,
		Crossing* crossings, size_t* turn, size_t rays_per_thread) {
	size_t thread_num = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;
	// is this okay? will the compiler optimize this? let's hope...
	size_t max_raynum = min(consts::NRAYS, (thread_num+1) * rays_per_thread);
	Xyz<double> k0 = {
		-1 * BEAM_NORM[beamnum][0],
		-1 * BEAM_NORM[beamnum][1],
		-1 * BEAM_NORM[beamnum][2]
	};
	Xyz<double>* child1 = child + (consts::NT * thread_num * 2);
	Xyz<double>* child2 = child1 + consts::NT;
	double* dist1 = dist + (consts::NT * thread_num * 2);
	double* dist2 = dist1 + consts::NT;
	for (size_t raynum = thread_num * rays_per_thread; raynum < max_raynum; raynum++) {
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
		start_pos.y = start_pos.y*cos(theta2) + tmp_x0*sin(theta2);

		// launch child rays
		constexpr double dist = 0.2 * consts::DX;
		Xyz<double> delta1 = {-k0.y, -k0.x, 0};
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
		launch_parent_ray(mesh, deden,
			child1, dist1, c1_size,
			child2, dist2, c2_size,
			start_pos, k0, crossings_ind, turn + b_mem_num*consts::NRAYS + raynum, TEMPinit_intensity);
	}
}

__device__ size_t launch_parent_ray(MeshPoint* mesh, Xyz<double>* deden,
		Xyz<double>* child1, double* dist1, size_t c1_size,
		Xyz<double>* child2, double* dist2, size_t c2_size,
		Xyz<double> pos, Xyz<double> k0, Crossing* crossings, size_t* turn, double TEMPintensity) {
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
			pow(ab.x * ac.y - ab.y * ac.x, 2));

	Xyz<size_t> mesh_pos = get_mesh_coords(mesh, pos);

	double k = get_k(mesh, pos, mesh_pos);
	double knorm = mag(k0);
	Xyz<double> vel = {
		pow(consts::C_SPEED, 2) * ((k0.x / knorm) * k) / consts::OMEGA,
		pow(consts::C_SPEED, 2) * ((k0.y / knorm) * k) / consts::OMEGA,
		pow(consts::C_SPEED, 2) * ((k0.z / knorm) * k) / consts::OMEGA,
	};

	double curr_dist = 0.0;
	double kds = 0.0;
	//double phase = 0.0;
	double permittivity = 1 - get_pt(mesh, mesh_pos)->eden / consts::NCRIT;
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

		double prev_permittivity = permittivity;
		permittivity = 1 - get_pt(mesh, mesh_pos)->eden / consts::NCRIT;
		if (permittivity < 0.0) permittivity = 0.0;

		double c_perms[3];
		double c_distances[3]; // to sort by
		size_t this_cnum = 0;

		// 3 maximum crosses, then sorted at the end by their... boxes? x then y then z...?
		// closure that modifies the crossing at index cnum,
		// and then increments cnum.
		auto new_crossing = [&] (double x, double y, double z, double frac) {
			// first, get mesh pos of crossing (cross[xyz]Ind)
			Xyz<size_t> c_mesh_pos = get_mesh_coords(mesh, {x, y, z});
			crossings[cnum].pt = {x, y, z};
			crossings[cnum].boxes = c_mesh_pos;
			// then, compute localAreaRatio, localKds, localEnergy, localPermittivity
			double distance_to_crossing = sqrt(pow(x - prev_pos.x, 2) +
					pow(y - prev_pos.y, 2) + pow(z - prev_pos.z, 2));
			c_distances[this_cnum] = distance_to_crossing;
			Xyz<double> childp1 = interp_xyz(child1, dist1, c1_size,
					curr_dist + distance_to_crossing);
			Xyz<double> childp2 = interp_xyz(child2, dist2, c2_size,
					curr_dist + distance_to_crossing);

			Xyz<double> interpk = {
				frac * vel.x + (1-frac) * prev_vel.x,
				frac * vel.y + (1-frac) * prev_vel.y,
				frac * vel.z + (1-frac) * prev_vel.z
			};

			// "getProjectedTriangleArea" which takes xyz, childp1/2, interpk
			double kmag = mag(interpk);
			ab = {childp1.x - pos.x, childp1.y - pos.y, childp1.z - pos.z};
			ac = {childp2.x - pos.x, childp2.y - pos.y, childp2.z - pos.z};
			Xyz<double> tri_norm = { // called just [xyz]norm in Shuang's code
				ab.y * ac.z - ab.z * ac.y,
				ab.z * ac.x - ab.x * ac.z,
				ab.x * ac.y - ab.y * ac.x
			};
			//NOTE: Shuang's code has (area = 0.5 * normmag) * abs... / kmag / normmag
			//where normmag = norm(tri_norm)
			//I wrote 0.5 * abs... / kmag
			//There IS a difference when I tested but that is very small
			//(like one millionth of a percent.. but if things are broken,
			//maybe change this first!!)
			crossings[cnum].area_ratio = (0.5 * abs(
					tri_norm.x * interpk.x +
					tri_norm.y * interpk.y +
					tri_norm.z * interpk.z) / kmag)
				/ init_area;
			crossings[cnum].kds = exp(kds - distance_to_crossing *
					get_pt(mesh, c_mesh_pos)->kib_multiplier);
			crossings[cnum].i_b = TEMPintensity * crossings[cnum].kds;
			c_perms[this_cnum] = frac * permittivity + (1-frac) * prev_permittivity;
			// NOTE: all of the stuff about phase is commented out in the version
			// that Shuang gave me, but this is where it would go if it was here,
			// and this is the line from the 2D impl.
			// crossings[cnum].phase = phase + distance_to_crossing * meshpt.permittivity_multiplier;

			cnum++;
			this_cnum++;
		};

		if (mesh_pos.x != prev_mesh_pos.x) {
			double crossx = get_pt(mesh, mesh_pos)->pt.x;
			if (!((pos.x > crossx && prev_pos.x <= crossx) ||
						(pos.x < crossx && prev_pos.x >= crossx))) {
				crossx = get_pt(mesh, prev_mesh_pos)->pt.x;
			}
			new_crossing(
				crossx,
				prev_pos.y + // crossy
					((pos.y - prev_pos.y) / (pos.x - prev_pos.x) * (crossx - prev_pos.x)),
				prev_pos.z + // crossz
					((pos.z - prev_pos.z) / (pos.x - prev_pos.x) * (crossx - prev_pos.x)),
				(crossx - prev_pos.x) / (pos.x - prev_pos.x) // frac
			);
		}
		if (mesh_pos.y != prev_mesh_pos.y) {
			double crossy = get_pt(mesh, mesh_pos)->pt.y;
			if (!((pos.y > crossy && prev_pos.y <= crossy) ||
						(pos.y < crossy && prev_pos.y >= crossy))) {
				crossy = get_pt(mesh, prev_mesh_pos)->pt.y;
			}
			new_crossing(
				prev_pos.x + // crossx
					((pos.x - prev_pos.x) / (pos.y - prev_pos.y) * (crossy - prev_pos.y)),
				crossy,
				prev_pos.z + // crossz
					((pos.z - prev_pos.z) / (pos.y - prev_pos.y) * (crossy - prev_pos.y)),
				(crossy - prev_pos.y) / (pos.y - prev_pos.y) // frac
			);
		}
		if (mesh_pos.z != prev_mesh_pos.z) {
			double crossz = get_pt(mesh, mesh_pos)->pt.z;
			if (!((pos.z > crossz && prev_pos.z <= crossz) ||
						(pos.z < crossz && prev_pos.z >= crossz))) {
				crossz = get_pt(mesh, prev_mesh_pos)->pt.z;
			}
			new_crossing(
				prev_pos.x +
					((pos.x - prev_pos.x) / (pos.z - prev_pos.z) * (crossz - prev_pos.z)),
				prev_pos.y +
					((pos.y - prev_pos.y) / (pos.z - prev_pos.z) * (crossz - prev_pos.z)),
				crossz,
				(crossz - prev_pos.z) / (pos.z - prev_pos.z)
			);
		}

		// sort crosses
		// update maxturnamp based on arearatio and permittivity
		if (this_cnum == 3) {
			// https://stackoverflow.com/questions/4793251/sorting-int-array-with-only-3-elements
			// I used the second answer, which is more optimal... and ugly... SORRY!!!!
			// distances[0] <=> crossings[cnum-3]
			// distances[1] <=> crossings[cnum-2]
			// distances[2] <=> crossings[cnum-1]
			Crossing temp;
			double temp_perm;
			if (c_distances[0] < c_distances[1]) {
				if (c_distances[1] > c_distances[2]) {
					if (c_distances[0] < c_distances[2]) {
						swap(crossings[cnum-2], crossings[cnum-1]);
						swap(c_perms[1], c_perms[2]);
					} else {
						temp = crossings[cnum-3]; temp_perm = c_perms[0];
						crossings[cnum-3] = crossings[cnum-1]; c_perms[0] = c_perms[2];
						crossings[cnum-1] = crossings[cnum-2]; c_perms[2] = c_perms[1];
						crossings[cnum-2] = temp; c_perms[1] = temp_perm;
					}
				}
			} else {
				if (c_distances[1] < c_distances[2]) {
					if (c_distances[0] < c_distances[2]) {
						swap(crossings[cnum-3], crossings[cnum-2]);
						swap(c_perms[0], c_perms[1]);
					} else {
						temp = crossings[cnum-3]; temp_perm = c_perms[0];
						crossings[cnum-3] = crossings[cnum-2]; c_perms[0] = c_perms[1];
						crossings[cnum-2] = crossings[cnum-1]; c_perms[1] = c_perms[2];
						crossings[cnum-1] = temp; c_perms[2] = temp_perm;
					}
				} else {
					swap(crossings[cnum-3], crossings[cnum-1]);
					swap(c_perms[0], c_perms[2]);
				}
			}
		} else if (this_cnum == 2) {
			if (c_distances[0] > c_distances[1]) {
				swap(crossings[cnum-1], crossings[cnum-2]);
				swap(c_perms[0], c_perms[1]);
			}
		}
		for (size_t i = 1; i <= this_cnum; i++) {
			double ray_amp = 1 / crossings[cnum-i].area_ratio * c_perms[i-1];
			if (ray_amp > max_turn_amp) {
				max_turn_amp = ray_amp;
				i_max_turn_amp = cnum-i;
			}
			if (cnum-i-1 > 0)
				calc_dk(crossings + cnum-i-1, crossings + cnum-i);
		}
		// update marked (TODO)
		// update currDist, kds, totalEnergy based on step_size = distance travelled.
		double distance_travelled = sqrt(pow(pos.x - prev_pos.x, 2) +
				pow(pos.y - prev_pos.y, 2) +
				pow(pos.z - prev_pos.z, 2));
		curr_dist += distance_travelled;
		kds -= distance_travelled * get_pt(mesh, mesh_pos)->kib_multiplier;
	}
	return cnum;
}

__device__ void calc_dk(Crossing* cross, Crossing* cross_next) {
	Xyz<double> dk = {
		cross_next->pt.x - cross->pt.x,
		cross_next->pt.y - cross->pt.y,
		cross_next->pt.z - cross->pt.z
	};
	double dkmag = mag(dk);
	cross->dk = {dk.x/dkmag, dk.y/dkmag, dk.z/dkmag};
	cross->dkmag = dkmag * 10000.0;
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
