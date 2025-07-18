#include <fstream>
#include <cstddef>
#include <cmath>
#include <chrono>
#include <H5Cpp.h>
#include <iostream>
#include "consts.hpp"
#include "structs.hpp"
#include "utils.cuh"
#include "ray_trace.hpp"
#include "cbet.hpp"

using namespace std;

// Loads spherical mesh from files that are hard-coded in right now
// but probably shouldn't be?
MeshPoint* read_spherical_mesh() {
	double* r_data = new double[consts::NR];
	double* te_data = new double[consts::NR];
	double* ne_data = new double[consts::NR];
	double* mach_data = new double[consts::NR];
	double* nu_ei_c = new double[consts::NR];

	fstream te_file;
	fstream mach_file;
	fstream ne_file;
	te_file.open("./output_77066_te.txt");
	mach_file.open("./output_77066_mach.txt");
	ne_file.open("./output_77066_ne.txt");
	for (size_t i = 0; i < consts::NR; i++) {
		// i guess the first column of each file is the same...?
		te_file >> r_data[i] >> te_data[i];
		mach_file >> r_data[i] >> mach_data[i];
		ne_file >> r_data[i] >> ne_data[i];
	}
	te_file.close();
	mach_file.close();
	ne_file.close();

	// calculate nu_ei_c
	for (size_t i = 0; i < consts::NR; i++) {
		double log_lambda;
		double te = te_data[i] / 1000;
		if (te > 0.01 * pow(consts::ZEFF, 2)) {
			log_lambda = 6.68 + log(consts::LASER_WAVE_LENGTH * te);
		} else {
			log_lambda = 9.13 + log(consts::LASER_WAVE_LENGTH * pow(te, 1.5) / consts::ZEFF);
		}
		nu_ei_c[i] = ((0.102/1.88e3)*consts::ZEFF*log_lambda) /(consts::LASER_WAVE_LENGTH * pow(te, 1.5))  * consts::W0/1e12/2;
	}

	// create mesh
	MeshPoint* mesh = new MeshPoint[consts::GRID];

	// since mesh is same size in all directions and the zones are linearly
	// sized, little shortcut
	double dx = (consts::XMAX - consts::XMIN)/((double)consts::NX - 1.0);
	double* spanx = new double[consts::NX];
	for (size_t i = 0; i < consts::NX; i++) {
		spanx[i] = (double)i * dx + consts::XMIN;
	}

	for (size_t x = 0; x < consts::NX; x++) {
		for (size_t y = 0; y < consts::NY; y++) {
			for (size_t z = 0; z < consts::NZ; z++) {
				MeshPoint* pt = get_pt(mesh, x, y, z);
				pt->pt.x = spanx[x];
				pt->pt.y = spanx[y];
				pt->pt.z = spanx[z];

				double dist = sqrt(pow(pt->pt.x, 2) + pow(pt->pt.y, 2) + pow(pt->pt.z, 2));
				pt->eden = interp(ne_data, r_data, consts::NR, dist);
				double nu_ei_c_interp = interp(nu_ei_c, r_data, consts::NR, dist);
				pt->kib = 2 * nu_ei_c_interp * 1e12/(consts::C_SPEED * 1e4);
				pt->kib *= pow(pt->eden / consts::NCRIT, 2);
				pt->dielectric = 1 - fmin(0.99, pt->eden / consts::NCRIT);
				//pt->permittivity_multiplier = fmax(sqrt_dielectric, 0.0) * consts::OMEGA / consts::C_SPEED;
				double mach_over_dist = interp(mach_data, r_data, consts::NR, dist) / dist;
				pt->machnum.x = -pt->pt.x * mach_over_dist;
				pt->machnum.y = -pt->pt.y * mach_over_dist;
				pt->machnum.z = -pt->pt.z * mach_over_dist;
			}
		}
	}

	delete[] r_data;
	delete[] te_data;
	delete[] ne_data;
	delete[] mach_data;
	delete[] nu_ei_c;
	delete[] spanx;

	return mesh;
}

// Writes the following datasets to edep.hdf5:
// /wplot, /absorption, /Coordinate_x, /Coordinate_y, /Coordinate_z
void save_hdf5(MeshPoint* mesh, Crossing* crossings, size_t* turn) {
	H5::H5File file("edep.hdf5", H5F_ACC_TRUNC);
	H5::IntType datatype(H5::PredType::NATIVE_DOUBLE);
	datatype.setOrder(H5T_ORDER_LE);
	hsize_t dims[3] = {consts::NX, consts::NY, consts::NZ};
	H5::DataSpace dataspace(3, dims);
	// 1) compute wplot
	//   a) for each beam,
	//     i) sum intensities, and counts per box - FOR EACH SHEET (based on turn)
	//     ii) wplot += sum(pow(sheet/count per sheet, 2))
	//   b) wplot = sqrt wplot
	double* wplot = new double[consts::GRID](); // zeroed
	double* sheet1 = new double[consts::GRID];
	double* sheet2 = new double[consts::GRID];
	int* count1 = new int[consts::GRID];
	int* count2 = new int[consts::GRID];
	for (size_t beamnum = 0; beamnum < consts::NBEAMS; beamnum++) {
		fill_n(count1, consts::GRID, 0);
		fill_n(sheet1, consts::GRID, 0.0);
		fill_n(count2, consts::GRID, 0);
		fill_n(sheet2, consts::GRID, 0.0);
		for (size_t raynum = 0; raynum < consts::NRAYS; raynum++) {
			Crossing* cross = crossings + (beamnum * consts::NRAYS + raynum)*consts::NCROSSINGS;
			size_t cnum = 0;
			while (cross->i_b != 0) {
				if (cnum < turn[beamnum * consts::NRAYS + raynum]) {
					*(get_pt(sheet1, cross->boxes)) += cross->i_b;
					*(get_pt(count1, cross->boxes)) += 1;
				} else {
					*(get_pt(sheet2, cross->boxes)) += cross->i_b;
					*(get_pt(count2, cross->boxes)) += 1;
				}
				cross++;
				cnum++;
			}
		}
		for (size_t i = 0; i < consts::GRID; i++) {
			if (count1[i] > 0) wplot[i] += pow(sheet1[i] / count1[i], 2);
			if (count2[i] > 0) wplot[i] += pow(sheet2[i] / count2[i], 2);
		}
	}
	for (size_t i = 0; i < consts::GRID; i++) wplot[i] = sqrt(wplot[i]);
	H5::DataSet dataset = file.createDataSet("/wplot", datatype, dataspace);
	dataset.write(wplot, H5::PredType::NATIVE_DOUBLE);
	// 2) compute absorption
	// reusing wplot variable bc why not
	fill_n(wplot, consts::GRID, 0.0);
	for (size_t beamnum = 0; beamnum < consts::NBEAMS; beamnum++) {
		for (size_t raynum = 0; raynum < consts::NRAYS; raynum++) {
			Crossing* cross = crossings + (beamnum * consts::NRAYS + raynum) * consts::NCROSSINGS;
			while (cross->i_b != 0) {
				*(get_pt(wplot, cross->boxes)) += cross->absorption_data;
				cross++;
			}
		}
	}
	for (size_t i = 0; i < consts::GRID; i++) {
		wplot[i] /= pow(consts::DX, 3);
	}
	dataset = file.createDataSet("/absorption", datatype, dataspace);
	dataset.write(wplot, H5::PredType::NATIVE_DOUBLE);
	// 3) save coordinate plots
	// Why not just reuse x, y, z? (TEST THIS???)
	for (size_t i = 0; i < consts::GRID; i++) wplot[i] = mesh[i].pt.x;
	dataset = file.createDataSet("/Coordinate_x", datatype, dataspace);
	dataset.write(wplot, H5::PredType::NATIVE_DOUBLE);
	for (size_t i = 0; i < consts::GRID; i++) wplot[i] = mesh[i].pt.y;
	dataset = file.createDataSet("/Coordinate_y", datatype, dataspace);
	dataset.write(wplot, H5::PredType::NATIVE_DOUBLE);
	for (size_t i = 0; i < consts::GRID; i++) wplot[i] = mesh[i].pt.z;
	dataset = file.createDataSet("/Coordinate_z", datatype, dataspace);
	dataset.write(wplot, H5::PredType::NATIVE_DOUBLE);

	delete[] wplot;
	delete[] sheet1;
	delete[] count1;
	delete[] sheet2;
	delete[] count2;
}

int main() {
	printf("Running CBET with:\n\tNumber of rays per beam: %lu\n\tNumber of mesh zones: %lu (%lux%lux%lu)\n", consts::NRAYS, consts::GRID, consts::NX, consts::NY, consts::NZ);
	// Load data from files
	printf("Reading mesh\n");
	MeshPoint* mesh = read_spherical_mesh();

	printf("Allocating beam info on CPU\n");
	Crossing* crossings = new Crossing[consts::NBEAMS * consts::NRAYS * consts::NCROSSINGS]();
	size_t* turn = new size_t[consts::NBEAMS * consts::NRAYS];
	RaystorePt* raystore = new RaystorePt[consts::GRID * consts::NBEAMS]();
	printf("Starting ray tracing\n");
	ray_trace(mesh, crossings, turn, raystore);

	printf("Callng cbet\n");
	cbet(mesh, crossings, raystore);

	printf("Writing hdf5 file\n");
	save_hdf5(mesh, crossings, turn);

	delete[] crossings;
	delete[] turn;
	delete[] raystore;
	delete[] mesh;
	return 0;
}
