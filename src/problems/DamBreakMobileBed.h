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

#ifndef _DAMBREAKMOBILEBAD_H
#define	_DAMBREAKMOBILEBAD_H

#include "XProblem.h"
#include "Point.h"
#include "Rect.h"
#include "Cube.h"

class DamBreakMobileBed: public XProblem {
	private:
		double		hw, hs; // water and sediment height;
		double		lx, ly, lz; // reservoir dimensions
		double		zi; // interface vertical position
		double		effvisc_max;

	public:
		DamBreakMobileBed(GlobalData *);
		virtual void initializeParticles(BufferList &buffers, const uint numParticles);
};


#endif	/* _DAMBREAKMOBILEBAD_H */
