#ifndef CONSTS_H
#define CONSTS_H

#include <cmath>
#define _USE_MATH_DEFINES

// constexpr = compile-time constant
// I don't really know why I use it instead of const for most cases.
// But I'm sure there would be a good reason to if I could think of one.
namespace consts {
	// Mesh constants
	constexpr size_t NX = 500;
	constexpr double XMIN = -8.0e-4;
	constexpr double XMAX = 8.0e-4;

	constexpr size_t NZ = 500;
	constexpr double ZMIN = -8.0e-4;
	constexpr double ZMAX = 8.0e-4;

	constexpr double NU_EI_C = 500.0;

	// Beam constants
	constexpr size_t NBEAMS = 2;
	constexpr size_t NCROSSINGS = NX * 3; // TODO: Make a more educated calculation!!
	constexpr size_t RAYS_PER_ZONE = 4;
	constexpr double BEAM_MAX_Z = 3.0e-4;
	constexpr double BEAM_MIN_Z = -3.0e-4;

	constexpr size_t NRAYS = (size_t)((double)RAYS_PER_ZONE * (BEAM_MAX_Z-BEAM_MIN_Z)/((ZMAX-ZMIN)/((double)NZ-1.0)));
	constexpr double OFFSET1 = 0.1e-4;
	constexpr double OFFSET2 = 0.1e-4;
	// 3beam test case
	//constexpr double OFFSET2 = -4.0e-4;
	constexpr double OFFSET3 = 3.0e-4;
	constexpr double DIR1[] = {1.0, 0.0};
	constexpr double DIR2[] = {0.0, 1.0};
	constexpr double CHILD_OFFSET = 0.1e-4;

	// comments copied from original c++ impl.
	constexpr double SIGMA = 2.0e-4;
	constexpr double INTENSITY = 1e17; // intensity of the beam in W/cm^2
	constexpr double COURANT_MULT = 0.1; // 0.37 // 0.25 // 0.36 // 0.22
	constexpr double URAY_MULT = INTENSITY*COURANT_MULT*(1.0/(double)RAYS_PER_ZONE);
	constexpr double URAY_MULT2 = 2.5*INTENSITY*COURANT_MULT*(1.0/(double)RAYS_PER_ZONE);

	// Ray tracing constants
	constexpr size_t NT = (size_t)((1.0 / COURANT_MULT) * (double)std::max(NX, NZ) * 2.0);
	constexpr double DX = (XMAX-XMIN)/((double)NX-1.0);
	constexpr double DZ = (ZMAX-ZMIN)/((double)NZ-1.0);
	// hoisted up from scientific constants section
	constexpr double C_SPEED = 29979245800.0; // speed of light in cm/s
	constexpr double DT = COURANT_MULT*std::max(DX, DZ)/C_SPEED;

	// CBET constants
	constexpr double MAX_INCR = 0.2;
	constexpr double CONVERGE = 1e-7;
	constexpr double CBETCONVERGENCE = 0.9990;

	// ===
	// scientific constants copied from def.h, comments copied also haha
	// ===
	// TODO: consider adding a C_SPEED^2 const
	constexpr double E0 = 8.85418782e-12; // permittivity of free space in m^-3 kg^-1 s^4 A^2
	constexpr double ME = 9.10938356e-31; // electron mass in kg
	constexpr double EC = 1.60217662e-19; // electron charge in C

	constexpr double LAMBDA = 1.053e-4/3.0; // wavelength of light, in cm. This is frequency-tripled "3w" or "blue" (UV) light
	constexpr double FREQ = C_SPEED/LAMBDA; // frequency of light, in Hz
	constexpr double OMEGA = 2.0*M_PI*FREQ; // frequency of light, in rad/s
	// the critical density occurs when omega = omega_p,e
	// NOTE: replaced pow(omega,2) with omega*omega and pow(ec, 2) with ec*ec
	constexpr double NCRIT = 1e-6*(OMEGA*OMEGA*ME*E0/(EC*EC));

	// More CBET constants
	constexpr double ESTAT = 4.80320427e-10; // electron charge in statC
	constexpr double ME_G = 9.10938356e-28; // electron mass in g
	constexpr double KB = 1.3806485279e-16; // Boltzmann constant in erg/K
	constexpr double NORM = 1e14;
	constexpr double CBET_CONST = (8.0*M_PI*1e7*NORM/C_SPEED)*(ESTAT*ESTAT/(4.0*ME_G*C_SPEED*KB*1.1605e7))*1e-4;

	constexpr double Z = 3.1; // ionization state
	constexpr double TE_EV = 2.0e3; // Te_eV; the comment on Te is "Temperature of electron in K"
	constexpr double TI_EV = 1.0e3; // Ti_eV; the comment on Ti is "Temperature of ion in K"
	constexpr double MI_KG = 10230.0*ME; // Mass of ion in kg
	// not a real const bc of sqrt
	const double CS = 1e2*std::sqrt(EC*(Z*TE_EV+3.0*TI_EV)/MI_KG);
	constexpr double IAW = 0.542940629585429; // ion-acoustic wave energy-damping rate (nu_ia/omega_s)!!
}

#endif
