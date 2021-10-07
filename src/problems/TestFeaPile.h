/*  Copyright (c) 2019 INGV, EDF, UniCT, JHU, NU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA
    Northwestern University, Evanston (IL), USA

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

#ifndef _TESTFEAPILE_H
#define	_TESTFEAPILE_H

#include "XProblem.h"
#include "Point.h"
#include "Cube.h"
#include "Cylinder.h"
#include "Rect.h"
#include "Vector.h"


class TestFeaPile: public XProblem {
	private:
		double		H;		// still water level
		double		lx, ly, lz;		// dimension of experiment box

		bool		need_write(double) const;

	public:
		TestFeaPile(GlobalData *);
		float3 ext_force_callback(const double t);
		void initializeParticles(BufferList &buffer, const uint numParticles);
};
#endif	/* _TESTFEAPILE_H */

