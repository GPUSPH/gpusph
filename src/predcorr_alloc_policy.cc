/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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
 * Predictor-corrector buffer allocation policy implementation
 */

#include "predcorr_alloc_policy.h"

#include "define_buffers.h"

//! All double buffers (for the predictor-corrector integration scheme)
/*! Presently, these are all the buffers holding particle properties,
 * except for the INFO buffer. The reason why the INFO buffer isn't
 * included is that it is only ever updated in-place (and rarely at that),
 * and in contrast to e.g. VERTICES (which is also only updated in-place)
 * it doesn't need a double buffer for sorting, since it is used as a sort key.
 */
#define BUFFERS_ALL_DBL (PARTICLE_PROPS_BUFFERS & ~BUFFER_INFO)

size_t
PredCorrAllocPolicy::get_max_buffer_count(flag_t Keys) const
{ return (Keys & BUFFERS_ALL_DBL ? 2 : 1); }

size_t
PredCorrAllocPolicy::get_buffer_count(flag_t Key) const
{ return (Key & BUFFERS_ALL_DBL ? 2 : 1); }

flag_t
PredCorrAllocPolicy::get_multi_buffered(flag_t Keys) const
{ return (Keys & BUFFERS_ALL_DBL); }
