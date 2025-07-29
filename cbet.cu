#include <cstdio>
#include <chrono>
#include <cub/cub.cuh>
#include <cuda.h>
#include "cbet.hpp"
#include "cbet.cuh"
#include "consts.hpp"
#include "utils.cuh"
#include "structs.hpp"

#define THREADS_PER_BLOCK 512

using namespace std::literals; // for dividing times by 1.0s

void cbet(MeshPoint* mesh, Crossing* crossings, RaystorePt* raystore) {
	/*printf("\tAllocating space for CBET, not counted in time\n");
	// TODO: make this use less memory by getting the minimum possible ncrossings value
	// (and maybe even the minimum possible nbeams value for the second nbeams in cbet_crosses?)
	double* coupling_mults = new double[consts::NBEAMS*consts::NRAYS*consts::NCROSSINGS*consts::NBEAMS]();
	double* w_mult_values = new double[consts::NBEAMS*consts::NRAYS*consts::NCROSSINGS]();*/

	auto start_time = std::chrono::high_resolution_clock::now();

	printf("\tAllocating and copying to GPU\n");

	MeshPoint* cuda_mesh;
	Crossing* cuda_crossings;
	RaystorePt* cuda_raystore;
	double* cuda_w_mult_values;
	CmultBounds* cuda_cmult_bounds;
	Cmult* cuda_cmults;

	gpuErrchk(cudaMalloc(&cuda_mesh, sizeof(MeshPoint) * consts::GRID));
	gpuErrchk(cudaMemcpy(cuda_mesh, mesh, sizeof(MeshPoint) * consts::GRID, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&cuda_crossings, sizeof(Crossing) * consts::NCROSSINGS * consts::NRAYS * consts::NBEAMS));
	gpuErrchk(cudaMemcpy(cuda_crossings, crossings, sizeof(Crossing) * consts::NCROSSINGS * consts::NRAYS * consts::NBEAMS, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&cuda_raystore, sizeof(RaystorePt) * consts::GRID * consts::NBEAMS));
	gpuErrchk(cudaMemcpy(cuda_raystore, raystore, sizeof(RaystorePt) * consts::GRID * consts::NBEAMS, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&cuda_w_mult_values, sizeof(double) * consts::NBEAMS * consts::NRAYS * consts::NCROSSINGS));
	gpuErrchk(cudaMemset(cuda_w_mult_values, 0, sizeof(double) * consts::NBEAMS * consts::NRAYS * consts::NCROSSINGS));
	gpuErrchk(cudaMalloc(&cuda_cmult_bounds, sizeof(CmultBounds) * consts::NBEAMS * consts::NRAYS * consts::NCROSSINGS));
	gpuErrchk(cudaDeviceSynchronize());

	printf("\tGetting the number of cmults needed\n");

	size_t num_cmults_needed;
	size_t* cuda_num_cmults;
	gpuErrchk(cudaMalloc(&cuda_num_cmults, sizeof(size_t)));
	gpuErrchk(cudaMemset(cuda_num_cmults, 0, sizeof(size_t)));
	get_num_cmults
		<<<CEIL_DIV(consts::NRAYS * consts::NBEAMS, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>
		(cuda_crossings, cuda_raystore, cuda_num_cmults);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(&num_cmults_needed, cuda_num_cmults, sizeof(size_t), cudaMemcpyDeviceToHost));

	// now let's see how many we can actually fit
	size_t gpu_bytes_free;
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemGetInfo(&gpu_bytes_free, NULL));
	size_t num_cmults = gpu_bytes_free / sizeof(Cmult);
	num_cmults -= num_cmults % consts::NRAYS * consts::NBEAMS; // trunc
	size_t num_cmults_per_ray = num_cmults / (consts::NRAYS * consts::NBEAMS);
	num_cmults_per_ray = min(num_cmults_per_ray, num_cmults_needed);
	num_cmults = num_cmults_per_ray * consts::NRAYS * consts::NBEAMS;
	printf("\tCalculating %lu cmults, which is %lu per ray, out of a maximum required %lu cmults per ray\n", num_cmults, num_cmults_per_ray, num_cmults_needed);
	gpuErrchk(cudaMalloc(&cuda_cmults, sizeof(Cmult) * num_cmults));
	gpuErrchk(cudaMemset(cuda_cmults, 0, sizeof(Cmult) * num_cmults));


	calc_coupling_mults
		<<<CEIL_DIV(consts::NRAYS * consts::NBEAMS, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>
		(cuda_mesh, cuda_crossings, cuda_raystore,
		 cuda_cmult_bounds, cuda_cmults, num_cmults_per_ray);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	printf("\tRunning CBET loop\n");
	double updateconv;
	double* cuda_updateconv;
	gpuErrchk(cudaMalloc(&cuda_updateconv, sizeof(double)));
	gpuErrchk(cudaMemset(cuda_updateconv, 0, sizeof(double)));

	double currmax = consts::MAX_INCR;
	size_t i;
	for (i = 1; i <= 500; i++) {
		printf("\t\tIteration %lu\n", i);
		get_cbet_gain
			<<<CEIL_DIV(consts::NRAYS * consts::NBEAMS, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>
			(cuda_mesh, cuda_crossings, cuda_raystore, cuda_w_mult_values,
			 cuda_cmult_bounds, cuda_cmults, num_cmults_per_ray);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		gpuErrchk(cudaMemset(cuda_updateconv, 0, sizeof(double)));
		update_intensities
			<<<CEIL_DIV(consts::NRAYS * consts::NBEAMS, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>
			(cuda_crossings, cuda_raystore,
			cuda_w_mult_values, currmax, cuda_updateconv);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		gpuErrchk(cudaMemcpy(&updateconv, cuda_updateconv, sizeof(double), cudaMemcpyDeviceToHost));
		printf("\t\tupdateconv: %lf\n", updateconv);
		if (updateconv <= consts::CONVERGE) break;
		double currmaxa = consts::MAX_INCR*pow(consts::CBETCONVERGENCE, i);
		double currmaxb = consts::CBETCONVERGENCE*updateconv;
		currmax = fmin(currmaxa, currmaxb);
	}
	printf("\tIterated %lu times, running post\n", i-1);

	gpuErrchk(cudaMemcpy(crossings, cuda_crossings, sizeof(Crossing) * consts::NCROSSINGS * consts::NRAYS * consts::NBEAMS, cudaMemcpyDeviceToHost));
	post(mesh, crossings);

	gpuErrchk(cudaFree(cuda_updateconv));
	gpuErrchk(cudaFree(cuda_mesh));
	gpuErrchk(cudaFree(cuda_crossings));
	gpuErrchk(cudaFree(cuda_raystore));
	gpuErrchk(cudaFree(cuda_w_mult_values));

	auto end_time = std::chrono::high_resolution_clock::now();
	printf("\tTotal time: %Lf seconds\n", (end_time - start_time) / 1.0s);
}

// gets the number of cmults needed per ray (max of all rays)
__global__ void get_num_cmults(Crossing* crossings, RaystorePt* raystore, size_t* num_cmults) {
	size_t thread_num = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;
	__shared__ size_t block_crossings[THREADS_PER_BLOCK];
	block_crossings[threadIdx.x] = 0;
	if (thread_num > consts::NBEAMS * consts::NRAYS) return;
	size_t ncrossings = 0;
	Crossing* cross = crossings + (thread_num)*consts::NCROSSINGS;
	while (cross->i_b != 0) {
		for (size_t o_b_num = 0; o_b_num < consts::NBEAMS; o_b_num++) {
			if (o_b_num == thread_num / consts::NRAYS) continue;
			RaystorePt* pt = get_pt(raystore + consts::GRID * o_b_num, cross->boxes);
			if (pt->cnum != 0 && pt->raynum != 0) ncrossings++;
		}
		cross++;
	}
	block_crossings[threadIdx.x] = ncrossings;
	using BlockReduce = cub::BlockReduce<size_t, THREADS_PER_BLOCK>;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	ncrossings = BlockReduce(temp_storage).Reduce(block_crossings, Max<size_t>{});
	if (threadIdx.x == 0) *num_cmults = max(ncrossings, *num_cmults);
}

__global__ void get_cbet_gain(MeshPoint* mesh, Crossing* crossings,
		RaystorePt* raystore, double* w_mult_values,
		CmultBounds* cmult_bounds, Cmult* cmults, size_t num_cmults_per_ray) {
	size_t thread_num = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;
	if (thread_num > consts::NBEAMS * consts::NRAYS) return;

	size_t ind_offset = (thread_num)*consts::NCROSSINGS;
	Crossing* cross = crossings + ind_offset;
	double* w_mult = w_mult_values + ind_offset;
	CmultBounds* bounds = cmult_bounds + ind_offset;
	cmults += thread_num * num_cmults_per_ray;
	while (cross->i_b != 0) {
		double cbet_sum = 0.0;
		// if calculated
		if (bounds->start <= num_cmults_per_ray) {
			for (size_t i = 0; i < bounds->size; i++) {
				size_t o_b_num = cmults[i + bounds->start].beamnum;
				RaystorePt* pt = get_pt(raystore + consts::GRID * o_b_num, cross->boxes);
				Crossing* raycross = crossings + (o_b_num*consts::NRAYS + pt->raynum)*consts::NCROSSINGS + pt->cnum;
				double avg_intensity = (raycross+1)->i_b > 0 ?
					(raycross->i_b+(raycross+1)->i_b)/2 :
					raycross->i_b;
				cbet_sum += cmults[i + bounds->start].coupling_mult * avg_intensity;
			}
		} else {
			for (size_t o_b_num = 0; o_b_num < consts::NBEAMS; o_b_num++) {
				if (o_b_num == thread_num / consts::NRAYS) continue;
				RaystorePt* pt = get_pt(raystore + consts::GRID * o_b_num, cross->boxes);
				if (pt->cnum == 0 && pt->raynum == 0) continue;
				Crossing* raycross = crossings + (o_b_num*consts::NRAYS + pt->raynum)*consts::NCROSSINGS + pt->cnum;
				double avg_intensity = (raycross+1)->i_b > 0 ?
					(raycross->i_b+(raycross+1)->i_b)/2 :
					raycross->i_b;
				double coupling_mult = get_coupling_mult(mesh, cross, raycross);
				cbet_sum += coupling_mult * avg_intensity;
			}
		}
		*w_mult = exp(-1.0*cbet_sum) * cross->absorb_coeff;

		cross++;
		bounds++;
		w_mult++;
	}
}

__global__ void update_intensities(Crossing* crossings, RaystorePt* raystore,
		double* w_mult_values, double curr_max, double* updateconv) {
	size_t thread_num = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;
	__shared__ double conv_values[THREADS_PER_BLOCK];
	conv_values[threadIdx.x] = 0;
	if (thread_num > consts::NBEAMS * consts::NRAYS) return;

	double conv_max = 0.0;
	size_t ind_offset = (thread_num)*consts::NCROSSINGS;
	Crossing* cross = crossings + ind_offset;
	double* lcl_w_mult = w_mult_values + ind_offset;
	double i0 = cross->i_b;
	double mult_acc = 1.0;
	size_t cnum = 1;
	lcl_w_mult++;
	cross++;
	while (cross->i_b != 0) {
		double new_intensity = limit_energy(cross->i_b, i0, mult_acc, curr_max, &conv_max);
		mult_acc *= *lcl_w_mult;
		cross->i_b = new_intensity;
		cross++;
		lcl_w_mult++;
		cnum++;
	}
	conv_values[threadIdx.x] = conv_max;

	using BlockReduce = cub::BlockReduce<double, THREADS_PER_BLOCK>;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	conv_max = BlockReduce(temp_storage).Reduce(conv_values, Max<double>{});
	if (threadIdx.x == 0 && conv_max > *updateconv) *updateconv = conv_max;
}

__device__ double limit_energy(double i_prev, double i0, double mult_acc, double curr_max, double* max_change) {
	double i_curr = i0*mult_acc;
	// the fractional change in energy from imposing the update as is
	double fractional_change = abs(i_curr-i_prev)/i_prev;
	// update the convergence check variable
	*max_change = fmax(fractional_change, *max_change);
	// if the fractional change is too large, clamp the value
	if (fractional_change > curr_max) {
		int sign = (i_curr - i_prev > 0) ? 1 : -1;
		double correction = 1 + curr_max*sign;
		i_curr = i_prev*correction;
	}
	return i_curr;
}

__global__ void calc_coupling_mults(MeshPoint* mesh, Crossing* crossings, RaystorePt* raystore,
		CmultBounds* cmult_bounds, Cmult* cmults, size_t num_cmults_per_ray) {
	size_t thread_num = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;
	if (thread_num > consts::NRAYS*consts::NBEAMS) return;

	Crossing* cross = crossings + thread_num*consts::NCROSSINGS;
	CmultBounds* bounds = cmult_bounds + thread_num*consts::NCROSSINGS;
	Cmult* cmult_ptr = cmults + thread_num*num_cmults_per_ray;
	size_t cmult_num = 0;

	while (cross->i_b != 0 && cmult_num < num_cmults_per_ray) {
		bounds->start = cmult_num;
		for (size_t o_b_num = 0; o_b_num < consts::NBEAMS; o_b_num++) {
			if (o_b_num == thread_num / consts::NRAYS) continue;
			RaystorePt* pt = get_pt(raystore + consts::GRID * o_b_num, cross->boxes);
			if (pt->cnum == 0 && pt->raynum == 0) continue;
			cmult_ptr->coupling_mult = get_coupling_mult(
				mesh, cross,
				crossings + (o_b_num*consts::NRAYS + pt->raynum)*consts::NCROSSINGS + pt->cnum);
			cmult_ptr->beamnum = o_b_num;
			cmult_num++;
			cmult_ptr++;
			if (cmult_num >= num_cmults_per_ray) {
				// so it knows all need to be calculated
				bounds->start = num_cmults_per_ray+1;
				break;
			}
		}
		bounds->size = cmult_num - bounds->start;

		cross++;
		bounds++;
	}
	// make sure rest are blanked
	while (cross->i_b != 0) {
		bounds->start = num_cmults_per_ray+1;
		bounds++;
		cross++;
	}
}

// for one coupling multiplier
__device__ double get_coupling_mult(MeshPoint* mesh, Crossing* cross, Crossing* raycross) {
	// copied in wholesale from Shuang
	MeshPoint* mesh_pt = get_pt(mesh, cross->boxes);
	double area_avg = (raycross+1)->i_b != 0 ?
		(raycross->area_ratio+(raycross+1)->area_ratio)/2.0 :
		raycross->area_ratio;
	double ne_over_nc = mesh_pt->eden;
	if (ne_over_nc > consts::NCRIT) ne_over_nc = 0.99;
	else ne_over_nc = ne_over_nc / consts::NCRIT;
	double ne_over_nc_corrected = fmin(ne_over_nc, 1.0); // TODO can remove?
	double ne_term = sqrt(1 - ne_over_nc_corrected);
	double epsilon_eff = ne_term * ne_term;
	double interaction_mult = 1/(area_avg*ne_term)*1/sqrt(epsilon_eff);

	Xyz<double> k_seed = cross->dk;
	Xyz<double> k_pump = raycross->dk;

	double omega1 = consts::OMEGA, omega2 = consts::OMEGA;

	Xyz<double> iaw_vector = {
		(omega1*k_seed.x - omega2*k_pump.x)*sqrt(1-ne_over_nc)/consts::C_SPEED,
		(omega1*k_seed.y - omega2*k_pump.y)*sqrt(1-ne_over_nc)/consts::C_SPEED,
		(omega1*k_seed.z - omega2*k_pump.z)*sqrt(1-ne_over_nc)/consts::C_SPEED
	};
	double k_iaw = mag(iaw_vector);
	double eta_numerator = omega1-omega2 -
		(iaw_vector.x * mesh_pt->machnum.x +
		 iaw_vector.y * mesh_pt->machnum.y + 
		 iaw_vector.z * mesh_pt->machnum.z); // * consts::CS
	double eta_denominator = k_iaw; // * consts::CS
	double eta = eta_numerator/eta_denominator;

	// THIS ONE IS CONSTANT TODO MOVE TO CONSTS...
	double param1 = consts::CBET_CONST / (consts::OMEGA*(consts::TE_EV/1e3 + 3.0 * consts::TI_EV/1e3/consts::Z));
	double param2 = ne_over_nc/consts::IAW*consts::IAW*consts::IAW*eta;
	double param3 = pow(eta*eta-1.0, 2) + consts::IAW*consts::IAW*eta*eta;
	double param4 = interaction_mult;

	double coupling_mult = param1*param2/param3*param4*cross->dkmag;
	// Random polarization
	coupling_mult *= (1 + pow(k_seed.x * k_pump.x + k_seed.y * k_pump.y + k_seed.z * k_pump.z, 2)) / 4;
	
	return coupling_mult;
}

void post(MeshPoint* mesh, Crossing* crossings) {
	const double norm_factor_const = sqrt(8.0*M_PI/consts::C_SPEED) * consts::ESTAT / (consts::ME_G * consts::C_SPEED * consts::W0) * sqrt(1e14 * 1e7);
	for (size_t beamnum = 0; beamnum < consts::NBEAMS; beamnum++) {
		for (size_t raynum = 0; raynum < consts::NRAYS; raynum++) {
			Crossing* cross = crossings + ((beamnum*consts::NRAYS) + raynum) * consts::NCROSSINGS;
			size_t cnum = 0;
			while (cross->i_b != 0) {
				double area_avg = (cross+1)->i_b != 0 ?
					(cross->area_ratio+(cross+1)->area_ratio)/2.0 :
					cross->area_ratio;
				double ne_over_nc = get_pt(mesh, cross->boxes)->eden;
				if (ne_over_nc > consts::NCRIT) ne_over_nc = 9.04e21; // hard coded??
				ne_over_nc = ne_over_nc / consts::NCRIT;
				double ne_over_nc_corrected = fmin(ne_over_nc, 1.0); // TODO can remove?
				double ne_term = sqrt(1 - ne_over_nc_corrected);
				double epsilon_eff = ne_term * ne_term;
				double interaction_mult = 1/(area_avg*ne_term)*1/sqrt(epsilon_eff);
				double norm_factor = norm_factor_const * sqrt(interaction_mult) * pow(epsilon_eff, 0.25);
				double prev_intensity = cross->i_b;
				cross->i_b = sqrt(prev_intensity) * norm_factor;

				double absorption = cnum < 2 ? 0 : (1 - cross->kds / (cross-1)->kds);
				// todo: why?
				double power = prev_intensity * 1e2 * pow((consts::BEAM_MAX_Z-consts::BEAM_MIN_Z) / (consts::NRAYS_X-1), 2.0);
				cross->absorption_data = absorption * power;

				cross++;
				cnum++; // used in absorption
			}
		}
	}
}
