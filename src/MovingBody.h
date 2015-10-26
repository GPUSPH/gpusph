/*  Copyright 2015 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#ifndef _MOVING_BODY_H
#define _MOVING_BODY_H

#include "vector_math.h"
#include "Object.h"

enum MovingBodyType {
	MB_ODE,
	MB_FORCES_MOVING,
	MB_MOVING
};

typedef struct KinematicData {
	double3			crot; ///< Center of rotation of the body
	double3			lvel; ///< Linear velocity
	double3			avel; ///< Angular velocity
	EulerParameters	orientation;

	KinematicData():
		crot(make_double3(0.0)),
		lvel(make_double3(0.0)),
		avel(make_double3(0.0)),
		orientation(EulerParameters())
	{};

	KinematicData(const KinematicData& kdata) {
		crot = kdata.crot;
		lvel = kdata.lvel;
		avel = kdata.avel;
		orientation = kdata.orientation;
	};

	KinematicData& operator = (const KinematicData& source) {
		crot = source.crot;
		lvel = source.lvel;
		avel = source.avel;
		orientation = source.orientation;
		return *this;
	};
} KinematicData;

typedef struct MovingBodyData {
	uint				index; ///< Sequential insertion index (NOTE: NOT index in the array)
	MovingBodyType		type;
	Object				*object;
	KinematicData		kdata;
	KinematicData		initial_kdata;

	MovingBodyData(): index(0), type(MB_MOVING), object(NULL), kdata(KinematicData()), initial_kdata(KinematicData()) {};

	MovingBodyData(const MovingBodyData& mbdata) {
		index = mbdata.index;
		type = mbdata.type;
		object = mbdata.object;
		kdata = mbdata.kdata;
		initial_kdata = mbdata.initial_kdata;
	};

	MovingBodyData& operator = (const MovingBodyData& source) {
		index = source.index;
		type = source.type;
		object = source.object;
		kdata = source.kdata;
		initial_kdata = source.initial_kdata;
		return *this;
	};
} MovingBodyData;

typedef std::vector<MovingBodyData *> MovingBodiesVect;

#endif

