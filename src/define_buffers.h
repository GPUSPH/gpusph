/*  Copyright 2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

/* Define one flag for each buffer which is used in a worker */

#ifndef DEFINED_BUFFERS
#define DEFINED_BUFFERS

// sanity check
#ifndef FIRST_DEFINED_BUFFER
#error "define_buffers.h was included without specifying starting point"
#endif

// start from FIRST_DEFINED_BUFFER
#define BUFFER_POS			FIRST_DEFINED_BUFFER
#define BUFFER_VEL			(BUFFER_POS << 1)
#define BUFFER_INFO			(BUFFER_VEL << 1)
#define BUFFER_HASH			(BUFFER_INFO << 1)

#define BUFFER_PARTINDEX	(BUFFER_HASH << 1)
#define BUFFER_INVINDEX		(BUFFER_PARTINDEX << 1)
#define BUFFER_CELLSTART	(BUFFER_INVINDEX << 1)
#define BUFFER_CELLEND		(BUFFER_CELLSTART << 1)

#define BUFFER_NEIBSLIST	(BUFFER_CELLEND << 1)

#define BUFFER_FORCES		(BUFFER_NEIBSLIST << 1)

#define BUFFER_XSPH			(BUFFER_FORCES << 1)

#define BUFFER_TAU			(BUFFER_XSPH << 1)

#define BUFFER_VORTICITY	(BUFFER_TAU << 1)
#define BUFFER_NORMALS		(BUFFER_VORTICITY << 1)

#define BUFFER_BOUNDELEMENTS	(BUFFER_NORMALS << 1)
#define BUFFER_GRADGAMMA		(BUFFER_BOUNDELEMENTS << 1)
#define BUFFER_VERTICES			(BUFFER_GRADGAMMA << 1)
#define BUFFER_PRESSURE			(BUFFER_VERTICES << 1)

#define BUFFER_TKE			(BUFFER_PRESSURE << 1)
#define BUFFER_EPSILON		(BUFFER_TKE << 1)
#define BUFFER_TURBVISC		(BUFFER_EPSILON << 1)
#define BUFFER_STRAIN_RATE	(BUFFER_TURBVISC << 1)
#define BUFFER_DKDE			(BUFFER_STRAIN_RATE << 1)

#define BUFFER_CFL			(BUFFER_DKDE << 1)
#define BUFFER_CFL_TEMP		(BUFFER_CFL << 1)
#define BUFFER_CFL_GAMMA	(BUFFER_CFL_TEMP << 1)
#define BUFFER_CFL_KEPS		(BUFFER_CFL_GAMMA << 1)

// last defined buffer. if new buffers are defined, remember to update this
#define LAST_DEFINED_BUFFER	BUFFER_CFL_KEPS


#endif

