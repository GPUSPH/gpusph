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
/*
 * File:   OdeObjects.h
 * Author: alexis
 *
 * Created on 9 juin 2012, 12:12
 */

#ifndef _ODEOBJECTS_H
#define	_ODEOBJECTS_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"
#include "Sphere.h"
#include "Cone.h"
#include "Torus.h"
#include "Cylinder.h"

#include "ode/ode.h"

class OdeObjects: public Problem {
	private:
		Cube		experiment_box;
		Cube		obstacle;
		PointVect	parts;
		PointVect	boundary_parts;
		PointVect	obstacle_parts;
		double		H;				// still water level
		double		lx, ly, lz;		// dimension of experiment box
		bool		wet;			// set wet to true have a wet bed experiment
		// ODE stuff
		Sphere		sphere;
		Cube		cube;
		Cylinder	cylinder;
		dGeomID		planes[5];
		dJointID	joint;


	public:
		OdeObjects(GlobalData *);
		virtual ~OdeObjects(void);

		int fill_parts(void);
		void copy_to_array(BufferList &);

		void ODE_near_callback(void *, dGeomID, dGeomID);

		void release_memory(void);
};
#endif	/* _ODEOBJECTS_H */

