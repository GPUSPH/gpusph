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

typedef struct {
	uint firstCell;		// inclusive
	uint lastCell;		// exclusive
	TransferDirection direction;
	TransferScope scope;
	uchar peer_gidx;	// global device index of sending/receiving peer
	uint firstParticle; // help caching the particle indices, which change after every bneeibs
	uint lastParticle;
} CellBurst;

typedef std::vector<CellBurst> BurstList;

#endif // _BURSTS_H


