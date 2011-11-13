/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

    This file is part of GPUSPH.

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

/* All viscosity variations. Include this after defining everything else */

#undef VISC_TYPE
#undef VISC_ARTVISC
#undef VISC_DYNAMIC
#undef VISC_KINEMATIC
#undef VISC_SPS

#define VISC_TYPE ARTVISC
#define VISC_ARTVISC
#define MIN_BLOCKS_FORCES 6
#include "forces_kernel.def"
#undef MIN_BLOCKS_FORCES
#undef VISC_TYPE
#undef VISC_ARTVISC

#define VISC_TYPE DYNAMICVISC
#define VISC_DYNAMIC
#define MIN_BLOCKS_FORCES 6
#include "forces_kernel.def"
#undef MIN_BLOCKS_FORCES
#undef VISC_TYPE
#undef VISC_DYNAMIC

#define VISC_TYPE KINEMATICVISC
#define VISC_KINEMATIC
#define MIN_BLOCKS_FORCES 6
#include "forces_kernel.def"
#undef MIN_BLOCKS_FORCES
#undef VISC_TYPE
#undef VISC_KINEMATIC

#define VISC_TYPE SPSVISC
#define VISC_SPS
#define MIN_BLOCKS_FORCES 4
#include "forces_kernel.def"
#undef MIN_BLOCKS_FORCES
#undef VISC_TYPE
#undef VISC_SPS

