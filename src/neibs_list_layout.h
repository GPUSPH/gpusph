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

#ifndef NEIBS_LIST_LAYOUT_H
#define NEIBS_LIST_LAYOUT_H

#include "backend_select.opt"

//! The neighbors list can be laid out in two ways:
//! * an interleaved neighbors list places one neighbor per particle in adjacent memory spaces,
//!   so that the next neighbor can be found numParticles ahead; this layout is optimal on GPU,
//!   since it allows work-items accessing the same neighbor index to load consecutive memory.
//! * a sequential neighbors list places all the neighbors to one particle first, followed by
//!   all the neighbors of the next particle and so on; this layout is optimal on CPU,
//!   since it improve caching during the neighbors list traversal

#if CPU_BACKEND_ENABLED
#define NL_INTERLEAVED 0
#else
#define NL_INTERLEAVED 1
#endif

#if NL_INTERLEAVED
#define ITH_STEP_NEIGHBOR(index, i, size)    (index + i)
#define ITH_NEIGHBOR(index, i, stride, size) (index + i*stride)
#else
#define ITH_STEP_NEIGHBOR(index, i, size)    ((index*size) + i)
#define ITH_NEIGHBOR(index, i, stride, size) ((index*size) + i)
#endif

#define ITH_NEIGHBOR_DEVICE(index, i) ITH_NEIGHBOR(index, i, d_neiblist_stride, d_neiblistsize)

#endif

