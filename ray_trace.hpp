#ifndef RAY_TRACE_H
#define RAY_TRACE_H

#include <tuple>
#include <cstddef>

size_t ray_trace(Mesh& mesh, Beam* beam, size_t nbeams);
size_t launch_child_ray(Ray& ray, Mesh& mesh, double* dedendx, double* dedendz, double* x, double* z, double* dist);
size_t launch_parent_ray(Ray& ray, Crossing* crossings, Mesh& mesh,
	double* dedendx, double* dedendz,
	double* childx, double* childz, double* distance, size_t child_size,
	size_t raynum);
double get_k(Mesh& mesh, double x0);
void calc_dk(Crossing& cross, Crossing& cross_next);

double interp(double* y, double* x, size_t len, double xp);
template<typename Func1, typename Func2>
double interp_closure(Func1 y, Func2 x, size_t len, double xp);

#endif
