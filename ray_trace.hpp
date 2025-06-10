#ifndef RAY_TRACE_HPP
#define RAY_TRACE_HPP
#include "structs.hpp"

struct RayTraceResult {
	Crossing** crossings;
	size_t** turn;
	size_t n_batches;
	size_t n_beams_per_batch;
};

void ray_trace(MeshPoint* mesh, RayTraceResult* result);
void free_rt_result(RayTraceResult* result);

#endif
