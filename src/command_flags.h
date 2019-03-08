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

/*! \file
 * Flags used as command parameters
 */

#ifndef _COMMAND_FLAGS_H
#define _COMMAND_FLAGS_H

#include "common_types.h"

//! \name Buffer key specifications
/*! Flags that indicate which buffer shuld be accessed for swaps, uploads, updates, etc.
 * These start from the next available bit from the bottom.
 * @{
 */

/*! Generic define for the beginning of the buffer keys, defined in
 * src/define_buffers.h
 */
#define FIRST_DEFINED_BUFFER	(flag_t(1))
/*! @} */

#endif

