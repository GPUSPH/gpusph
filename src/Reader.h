/*  Copyright (c) 2013-2019 INGV, EDF, UniCT, JHU

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
#ifndef _READER_H
#define _READER_H

#include <string>
#include <iostream>

#define CRIXUS_FLUID 1
#define CRIXUS_VERTEX 2
#define CRIXUS_BOUNDARY 3
#define CRIXUS_BOUNDARY_PARTICLE 4

struct ReadParticles {
	double Coords_0;
	double Coords_1;
	double Coords_2;
	double Normal_0;
	double Normal_1;
	double Normal_2;
	double Volume;
	double Surface;
	int ParticleType;
	int FluidType;
	int KENT;
	int MovingBoundary;
	int AbsoluteIndex;
	int VertexParticle1;
	int VertexParticle2;
	int VertexParticle3;
};

class Reader
{
protected:
	std::string		filename;
	size_t	npart;
public:
	Reader(void);
	virtual ~Reader(void);

	//! returns the number of particles in the input file
	virtual size_t getNParts(void) = 0;

	//! allocates the buffer and reads the data from the input file
	virtual void read(void) = 0;

	//! frees the buffer
	void empty(void);

	//! free the buffer, reset npart and filename
	void reset();

	//! sets the filename
	void setFilename(std::string const&);

	ReadParticles *buf;
};

#endif	/* _READER_H */
