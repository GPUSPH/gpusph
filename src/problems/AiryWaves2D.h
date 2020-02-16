/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

    This file is modified based on the Wave Tank problem of GPUSPH.
    First modified by Z. Wie, then by Morteza Derakhti derakhti@jhu.edu

*/

#ifndef _AIRYWAVES_H
#define	_AIRYWAVES_H

#include "XProblem.h"
#include "Point.h"
#include "Cube.h"
#include "Cylinder.h"
#include "Rect.h"
#include "Vector.h"


class AiryWaves2D: public XProblem {
	private:
		bool		use_cyl, use_bottom_plane;
		double		paddle_length;
		double		paddle_width;
		double		height, slope_length, beta;
		double		H, horizontal_flat;	// still water level, slope length	
		double		WavePeriod, WaveNumber, WaveHeight;
                double		stroke;	// paddle stroke length
		double		lx, ly, lz;		// dimension of experiment box
		bool		m_usePlanes; // use planes or boundaries
		Rect*		paddle;					
		// Moving boundary data
		double		paddle_amplitude, paddle_omega;
		double3         paddle_origin;
		double		paddle_tstart, paddle_tend;

	public:
		AiryWaves2D(GlobalData *);
		void copy_planes(PlaneList &);
		void fillDeviceMap();
		void moving_bodies_callback(const uint, Object*, const double, const double, const float3&,
									const float3&, const KinematicData &, KinematicData &,
									double3&, EulerParameters&);
		virtual void initializeParticles(BufferList &buffers, const uint numParticles);
		bool need_write(double t) const;
};
#endif	/* _AIRYWAVES2D_H */

