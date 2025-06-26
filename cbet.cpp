#include <cstdio>
#include "cbet.hpp"
#include "consts.hpp"
#include "utils.cuh"
#include "structs.hpp"

void cbet(MeshPoint* mesh, Crossing* crossings) {
	post(mesh, crossings);
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
				double ne_over_nc_corrected = fmin(ne_over_nc, 1.0);
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
