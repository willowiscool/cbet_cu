#ifndef CBET_HPP
#define CBET_HPP
#include "structs.hpp"

void cbet(MeshPoint* mesh, Crossing* crossings, RaystorePt* raystore);

void post(MeshPoint* mesh, Crossing* crossings);
void calc_coupling_mults(MeshPoint* mesh, Crossing* crossings, RaystorePt* raystore, double* coupling_mults);
double get_coupling_mult(MeshPoint* mesh, Crossing* cross, Crossing* raycross);
void get_cbet_gain(Crossing* crossings, RaystorePt* raystore,
		double* coupling_mults, double* w_mult_values);
double update_intensities(Crossing* crossings, RaystorePt* raystore,
		double* coupling_mults, double* w_mult_values, double curr_max);
double limit_energy(double i_prev, double i0, double mult_acc, double curr_max, double* max_change);
#endif
