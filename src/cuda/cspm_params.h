/*  Copyright (c) 2021 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

    GPUSPH is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GPUSPH is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GPUSPH.  If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file
 * Parameter structures for the CSPM kernel used by CCSPH and ANTUONO / DELTA_SPH
 */

#ifndef CSPM_PARAMS_H
#define CSPM_PARAMS_H

#include "common_params.h"
#include "cond_params.h"
#include "neibs_list_params.h"

struct delta_cspm_params
{
	float4 * __restrict__ renorm_dens_grad;

	delta_cspm_params(BufferList& bufwrite) :
		renorm_dens_grad(bufwrite.getData<BUFFER_RENORMDENS>())
	{}
};

template<BoundaryType boundarytype, DensityDiffusionType densitydiffusiontype_, flag_t simflags_,
	typename cond_delta = typename COND_STRUCT(densitydiffusiontype_ == ANTUONO, delta_cspm_params),
	typename cond_cspm =  typename COND_STRUCT(HAS_CCSPH(simflags_), cspm_params<true>)
	>
struct cspm_coeff_params : neibs_interaction_params<boundarytype>, cond_delta, cond_cspm
{
	static constexpr DensityDiffusionType densitydiffusiontype = densitydiffusiontype_;
	static constexpr flag_t simflags = simflags_;

	cspm_coeff_params(
		BufferList const& bufread,
		BufferList bufwrite,
		const	uint	numParticles,
		const	float	slength,
		const	float	influenceradius)
	:
		neibs_interaction_params<boundarytype>(bufread, numParticles, slength, influenceradius),
		cond_delta(bufwrite),
		cond_cspm(bufwrite)
	{}
};

#endif // CSPM_PARAMS_H

