/*  Copyright (c) 2019 INGV, EDF, UniCT, JHU

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

// Example usage of DEM with Problem API 1 aka XProblem

#include <string>

#include "DEMExample.h"
#include "cudasimframework.cu"

using namespace std;

DEMExample::DEMExample(GlobalData *gdata) : Problem(gdata)
{
	const int mlsIters = get_option("mls", 0); // --mls N to enable MLS filter every N iterations
	const int ppH = get_option("ppH", 16); // --ppH N to change deltap to H/N

	// density diffusion terms, see DensityDiffusionType
	const DensityDiffusionType rhodiff = get_option("density-diffusion", DENSITY_DIFFUSION_NONE);

	const string dem_file = get_option("dem", "half_wave0.1m.txt");

	// Use geometrical descriptions or implement walls and DEM with particles?
	const bool use_geometries = get_option("use-geometries", true);

	SETUP_FRAMEWORK(
		viscosity<ARTVISC>,
		boundary<LJ_BOUNDARY>
	).select_options(
		rhodiff,
		use_geometries, add_flags<ENABLE_DEM | ENABLE_PLANES>()
	);

	/* Simulation parameters */
	set_deltap(0.05);
	double water_height = 0.8;

	/* Physical parameters */
	setMaxFall(2.0);
	set_gravity(-9.81f);
	size_t water = add_fluid(1000.0f);
	set_equation_of_state(water, 7.0f, NAN /* autocompute from max fall */);

	/* Geometries */
	GeometryID dem = addDEM(dem_file, DEM_FMT_ASCII, use_geometries ? FT_NOFILL : FT_BORDER);
	GeometryID fluid_box = addDEMFluidBox(water_height);

	if (QUERY_ANY_FLAGS(simparams()->simflags, ENABLE_PLANES))
		vector<GeometryID> planes = addDEMPlanes();
	else if (simparams()->boundary_is_multilayer()) {
		// DEM boundaries start one layer out, so we need an extra deltap of margin
		// TODO FIXME this should be handled automatically
		addExtraWorldMargin(m_deltap);
	}


	add_writer(VTKWRITER, 0.1);
}

