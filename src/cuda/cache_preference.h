/*  Copyright (c) 2020 INGV, EDF, UniCT, JHU

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

#ifndef CUDA_CACHE_PREFERENCE_H
#define CUDA_CACHE_PREFERENCE_H

// On devices with compute capability 2.x, we want to distribute cache load
// between L1 cache and the texture cache. On older architectures (no L1 cache)
// we prefer using textures for all read-only arrays. Define PREFER_L1 to 1 or
// 0 accordingly. On 3.x, the L1 cache is only used for register spills, so
// exclude it from PREFER_L1. We keep PREFER_L1 on Maxwell because tests indicate
// that using textures leads to no improvement at best (and possibly some minor
// performance loss)

#if defined(__COMPUTE__)
#if __COMPUTE__ >= 20 && __COMPUTE__/10 != 3
#define PREFER_L1 1
#else
#define PREFER_L1 0
#endif
#endif

#endif
