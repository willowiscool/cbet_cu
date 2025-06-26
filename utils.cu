#include <cassert>
#include "consts.hpp"
#include "structs.hpp"
#include "utils.cuh"

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE //
#endif

using namespace std;

// panics instead of erroring... or undefined here in cpp?
HOST_DEVICE
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
