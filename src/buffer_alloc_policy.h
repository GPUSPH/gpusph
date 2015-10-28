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

#ifndef _BUFFER_ALLOC_POLICY_H
#define _BUFFER_ALLOC_POLICY_H

/* Buffer allocation policy: a class that determines, for each buffer,
 * how many copies are needed. This generally depends on the integration
 * scheme used, and potentially also at least the formulation (e.g.
 * a formulation where the particle volume evolves might need a double
 * buffer for the particle volume, while if a summation-density formulation
 * is used, a single copy of the particle volume buffer would suffice).
 */

#include "common_types.h"

struct BufferAllocPolicy {
	virtual ~BufferAllocPolicy() {};

	/* maximum number of copies needed for the given set of buffers
	 * (default: all buffers) */
	virtual size_t get_max_buffer_count(flag_t Keys = FLAG_MAX) const = 0;

	/* return the number of copies needed for buffer Key */
	virtual size_t get_buffer_count(flag_t Key) const = 0;

	/* return the (sub)set of buffers for which more than one copy is
	 * needed, among the ones requested */
	virtual flag_t get_multi_buffered(flag_t Keys = FLAG_MAX) const = 0;
};

#endif

