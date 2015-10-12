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

#ifndef _PREDCORR_ALLOC_POLICY_H
#define _PREDCORR_ALLOC_POLICY_H

#include "buffer_alloc_policy.h"

/* Predictor-corrector buffer allocation policy */

struct PredCorrAllocPolicy : public BufferAllocPolicy {
	virtual size_t get_max_buffer_count(flag_t Keys = FLAG_MAX) const;

	virtual size_t get_buffer_count(flag_t Key) const;

	virtual flag_t get_multi_buffered(flag_t Keys = FLAG_MAX) const;
};

#endif

