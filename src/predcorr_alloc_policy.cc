/*  Copyright 2014 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

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

#include "predcorr_alloc_policy.h"

#include "define_buffers.h"

// all double buffers (for the predictor-corrector integration scheme)
#define BUFFERS_ALL_DBL		(BUFFER_POS | BUFFER_VEL | BUFFER_INFO | \
	BUFFER_BOUNDELEMENTS | BUFFER_GRADGAMMA | BUFFER_VERTICES | \
	BUFFER_TKE | BUFFER_EPSILON | \
	BUFFER_TURBVISC | BUFFER_VOLUME | BUFFER_EULERVEL)

size_t
PredCorrAllocPolicy::get_max_buffer_count(flag_t Keys) const
{ return (Keys & BUFFERS_ALL_DBL ? 2 : 1); }

size_t
PredCorrAllocPolicy::get_buffer_count(flag_t Key) const
{ return (Key & BUFFERS_ALL_DBL ? 2 : 1); }

flag_t
PredCorrAllocPolicy::get_multi_buffered(flag_t Keys) const
{ return (Keys & BUFFERS_ALL_DBL); }
