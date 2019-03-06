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

#ifndef _WAVETANK_H
#define	_WAVETANK_H

#include "XProblem.h"


class WaveTank: public XProblem {
	private:
		bool		use_cyl, use_bottom_plane;
		double		paddle_length;
		double		paddle_width;
		double		h_length, height, slope_length, beta;
		double		H;		// still water level
		double		lx, ly, lz;		// dimension of experiment box

		// Moving boundary data
		double		paddle_amplitude, paddle_omega;
		double3		paddle_origin;
		double		paddle_tstart, paddle_tend;

	public:
		WaveTank(GlobalData *);
		void copy_planes(PlaneList &);

		void moving_bodies_callback(const uint, Object*, const double, const double, const float3&,
									const float3&, const KinematicData &, KinematicData &,
									double3&, EulerParameters&);
};
#endif	/* _WAVETANK_H */

