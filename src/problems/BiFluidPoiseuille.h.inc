/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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

#ifndef _BIFLUIDPOISEUILLE_H
#define	_BIFLUIDPOISEUILLE_H

#define PROBLEM_API 1
#include "Problem.h"

#include "Point.h"
#include "Rect.h"
#include "Cube.h"

class BIFLUIDPOISEUILLE_PROBLEM: public Problem {
	private:
		uint		dyn_layers;
		double3		dyn_offset;
		double3		margin;
		double		a, l;  // channel length (along X) and width (along Y)
		double		H, h; // channel height (along Z) and rounded height
		/* alpha is in [0; 1]
			*for alpha=1 --> only fluid 0
			*for alpha=0 --> only fluid 1
		*/
		double		alpha;
	size_t top, bottom; // fluid indices

	public:
		BIFLUIDPOISEUILLE_PROBLEM(GlobalData *);
		virtual void initializeParticles(BufferList &buffers, const uint numParticles);
};


#endif	/* _BIFLUIDPOISEUILLE_H */
