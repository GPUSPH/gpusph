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
 * File:   Seiche.h
 * Author: rad
 *
 * Created on February 11, 2011-2013, 2:48 PM
 */

#ifndef SEICHE_H
#define	SEICHE_H

#include "XProblem.h"

class Seiche: public XProblem {
	private:
		float		h, w, l;
		float		H; // still water level
		double		m_gtstart, m_gtend;

	public:
		Seiche(GlobalData *);

		void copy_planes(PlaneList &);
		float3 g_callback(const double);

};
#endif	/* SEICHE_H */

