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
 * Parameter structures for the CSPM kernel
 */

#ifndef CSPM_PARAMS_H
#define CSPM_PARAMS_H

#include "common_params.h"
#include "neibs_list_params.h"

template<BoundaryType boundarytype>
struct cspm_coeff_params : neibs_interaction_params<boundarytype>, cspm_params<true>
{
	cspm_coeff_params(
		BufferList const& bufread,
		BufferList bufwrite,
		const	uint	numParticles,
		const	float	slength,
		const	float	influenceradius)
	:
		neibs_interaction_params<boundarytype>(bufread, numParticles, slength, influenceradius),
		cspm_params<true>(bufwrite)
	{}
};

#endif // CSPM_PARAMS_H
