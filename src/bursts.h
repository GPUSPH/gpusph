/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#ifndef _BURSTS_H
#define _BURSTS_H

#include <vector>

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


