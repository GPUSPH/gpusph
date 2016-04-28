/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Universit� di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

  � This file is part of GPUSPH.

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
 * File:   SolitaryWave_04.h
 * Author: rad
 *
 * Created on February 7, 2011-2013, 2:39 PM
 */

#ifndef CSIDEVICE_H
#define	CSIDEVICE_H

#include <fstream>
#include "Problem.h"
#include "Point.h"
#include "Cube.h"
#include "Rect.h"
#include "Vector.h"
#include "Sphere.h"
#include "Torus.h"
#include "Cylinder.h"
#include "DataFileParser.h"


class CSIDevice: public Problem {
	private:
		DataFileParser	parser;
		Cube		experiment_box;
		int			i_use_bottom_plane;
		PointVect	parts;
		PointVect	boundary_parts;
		PointVect	paddle_parts;
        PointVect	test_points;

		double 		lx, ly, lz;		// Dimension of the computational domain
		double		h_length;		// Horizontal part of the experimental domain
		double		slope_length;	// Length of the inclined plane
		double		height;			// Still water (with z origin on the horizontal part)
		double		beta;			// Angle of the inclined plane
        double		H;				// still water level
		double		Hbox;			// height of experiment box
		Cube		paddle;
        double		paddle_length;
        double		paddle_width;
        double3		paddle_origin;
        double 		paddle_tstart;
        double		paddle_tend;
        double 		paddle_amplitude;
        double		paddle_omega;
    
		Cube		cube;
		dJointID	joint;

		Cube				platform;
		Cylinder			tower0, tower1, crossbar;
		Sphere				swing1, swing2;
        vector<dJointID> 	joints;
        double3				mooring[4];					// Mooring points on the sea bed
        double3				attachment_object_frame[4];	// Mooring points on the platform relative to the platform frame
        double3				attachment_rest_frame[4];	// Mooring points on the platform relative to rest frame
        double3				mooring_tension[4];			// Tension of the morring cables
        dJointFeedback 		jointFb;
        std::ofstream		jf_file;
        double 				chain_uw, chain_xdist;

	public:
		CSIDevice(GlobalData *);
		~CSIDevice(void);
		int fill_parts(void);

		void copy_to_array(BufferList &);
		void ODE_near_callback(void *, dGeomID, dGeomID);
		void moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
					const float3& force, const float3& torque, const KinematicData& initial_kdata,
					KinematicData& kdata, double3& dx, EulerParameters& dr);
		void bodies_forces_callback(const double t0, const double t1, const uint step, float3 *forces, float3 *torques);
		void writer_callback(CallbackWriter *cbw, uint numParts, BufferList const&, uint node_offset, double t,
				const bool testpoints) const;

	private:
		void release_memory(void);
		void build();
		double f(const double T, const double xdist, const double height);
		double dfdT(const double T, const double xdist);
		double dfdx(const double x, const double T);
		double find_tension(const double Ti, const double xdist, const double height);
};


#endif	/* CSIDevice_H */

