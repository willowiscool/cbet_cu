#include "consts.hpp"
#include "mesh.hpp"
#include "beam.hpp"
#include "ray_trace.hpp"
#include "main.hpp"
#include <iostream>
#include <string>
#include <H5Cpp.h>

int main() {
	cout << "Creating and initializing mesh" << endl;
	Mesh m = new_mesh();
	cout << "Creating and initializing beams" << endl;
	Beam b1 = beam1();
	Beam b2 = beam2();
	Beam beams[] = {b1, b2};
	cout << "Tracing rays" << endl;
	ray_trace(m, beams, consts::NBEAMS);

	cout << "Saving to hdf5" << endl;
	save_hdf5(m, beams, consts::NBEAMS, "out.hdf5");
	return 0;
}

// ***does not compute field amplitude!!! so implement that if needed!!!***
void save_hdf5(Mesh& mesh, Beam* beams, size_t nbeams, std::string filename) {
	double* wplot = new double[mesh.nx*mesh.nz]();

	for (size_t beamnum = 0; beamnum < nbeams; beamnum++) {
		// makes sure they're zeroed
		// but also like this is pretty not efficient?
		// but it happens only once per beam--it's fine
		double* intensities = new double[mesh.nx*mesh.nz]();
		size_t* ct = new size_t[mesh.nx*mesh.nz]();
		for (size_t cnum = 0; cnum < consts::NRAYS * consts::NCROSSINGS; cnum++) {
			Crossing& c = beams[beamnum].crossings[cnum];
			if (c.i_b != 0) {
				intensities[c.boxesx*mesh.nz + c.boxesz] += c.i_b;
				ct[c.boxesx*mesh.nz + c.boxesz] += 1;
			} else {
				// skip to next
				cnum += consts::NCROSSINGS - (cnum % consts::NCROSSINGS);
			}
		}
		for (size_t x = 0; x < mesh.nx; x++) {
			for (size_t z = 0; z < mesh.nz; z++) {
				wplot[x*mesh.nz + z] += pow(intensities[x*mesh.nz + z] / max(ct[x*mesh.nz + z], (size_t)1), 2);
			}
		}
		delete[] intensities;
		delete[] ct;
	}

	for (size_t x = 0; x < mesh.nx; x++) {
		for (size_t z = 0; z < mesh.nz; z++) {
			wplot[x*mesh.nz + z] = sqrt(wplot[x*mesh.nz + z]);
		}
	}

	H5::H5File file(filename, H5F_ACC_TRUNC);
	H5::IntType datatype(H5::PredType::NATIVE_DOUBLE);
	datatype.setOrder(H5T_ORDER_LE);
	hsize_t dims[2] = {mesh.nx, mesh.nz};
	H5::DataSpace dataspace(2, dims);
	H5::DataSet dataset = file.createDataSet("/wplot", datatype, dataspace);
	dataset.write(wplot, H5::PredType::NATIVE_DOUBLE);

	delete[] wplot;
}
