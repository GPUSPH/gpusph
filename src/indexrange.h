/*  Copyright (c) 2012-2019 INGV, EDF, UniCT, JHU

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
 * Index range data type.
 *
 * Defines information about the first/last index to contain specific information
 * (particles of a given type, particles to be processed, etc).
 */

#ifndef INDEX_RANGE_H
#define INDEX_RANGE_H

#include <limits.h>

#include "utils.h"

struct IndexRange
{
	unsigned int begin; // inclusive
	unsigned int end; // exclusive

	IndexRange() : begin(0), end(0) {}

	// no check for validity
	IndexRange(unsigned int begin_, unsigned int end_) : begin(begin_), end(end_) {}

	inline bool empty() const
	{ return end <= begin; }

	inline unsigned size() const
	{ return end - begin; }

	inline IndexRange intersect(IndexRange const& other) const
	{
		return IndexRange(
			// max begin, min end, without #include <algorithm>
			begin < other.begin ? other.begin : begin,
			end   > other.end   ? other.end   : end);
	}

	/// Returns the number of blocks needed to cover the range size, given the number of threads per block
	inline unsigned int numBlocks(unsigned int numThreads) const
	{ return div_up(size(), numThreads); }

	/// Returns the number of blocks needed to cover the range size, given the number of threads per block
	/// and a rounding up
	inline unsigned int numBlocks(unsigned int numThreads, unsigned int multiple_of) const
	{ return round_up(numBlocks(numThreads), multiple_of); }
};
#endif

