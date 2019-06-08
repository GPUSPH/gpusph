/*  Copyright (c) 2011-2018 INGV, EDF, UniCT, JHU

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
 * Data types to describe data transfer bursts
 */
#ifndef _BURSTS_H
#define _BURSTS_H

#include <vector>

#include "common_types.h"

typedef enum {SND, RCV} TransferDirection;

typedef enum {NODE_SCOPE, NETWORK_SCOPE} TransferScope;

typedef std::vector<uint> CellList;

typedef struct {
	// list of cells (linear indices)
	CellList cells;

	// global device index of sending/receiving peer
	uchar peer_gidx;
	// scope & direction (SND or RCV if NETWORK_SCOPE, only RCV for NODE_SCOPE)
	TransferDirection direction;
	TransferScope scope;

	// caching burst edges in terms of particle indices, after every buildneibs
	uint selfFirstParticle; // self p.index
	uint peerFirstParticle; // only useful for intra-node
	uint numParticles;
} CellBurst;

typedef std::vector<CellBurst> BurstList;

#endif // _BURSTS_H


