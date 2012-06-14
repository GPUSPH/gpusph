/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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

/* Simulation parameters for problems */

#ifndef _SIMPARAMS_H
#define _SIMPARAMS_H

typedef struct MbCallBack {
	short			type;
	float			tstart;
	float			tend;
	float3			origin;
	float3			vel;
	float3			disp;
	float			sintheta;
	float			costheta;
	float			omega;
	float			amplitude;
	float			phase;
} MbCallBack;


typedef struct SimParams {
	float			slength;			// smoothing length
	KernelType		kerneltype;			// kernel type
	float			kernelradius;		// kernel radius
	float			dt;					// initial timestep
	float			tend;				// simulation end time (0 means run forever)
	bool			xsph;				// true if XSPH correction
	bool			dtadapt;			// true if adaptive timestep
	float			dtadaptfactor;		// safety factor in the adaptive time step formula
	int				buildneibsfreq;		// frequency (in iterations) of neib list rebuilding
	int				shepardfreq;		// frequency (in iterations) of Shepard density filter
	int				mlsfreq;			// frequency (in iterations) of MLS density filter
	ViscosityType	visctype;			// viscosity type (1 artificial, 2 laminar)
	int				displayfreq;		// display update frequence (in seconds)
	int				savedatafreq;		// simulation data saving frequence (in displayfreq)
	int				saveimagefreq;		// screen capture frequence (in displayfreq)
	bool			mbcallback;			// true if moving boundary velocity varies
	bool			gcallback;			// true if using a variable gravity in problem
	bool			periodicbound;		// true in case of periodic boundary
	float			nlexpansionfactor;	// increase influcenradius by nlexpansionfactor for neib list construction
	bool			usedem;				// true if using a DEM
	SPHFormulation	sph_formulation;	// formulation to use for density and pressure computation
	BoundaryType	boundarytype;		// boundary force formulation (Lennard-Jones etc)
	bool			vorticity;			// true if we want to save vorticity
	bool            testpoints;         // true if we want to find velocity at testpoints
	bool            savenormals;        // true if we want to save the normals at free surface
	bool            surfaceparticle;    // true if we want to find surface particles
	//WaveGage
	bool			writeWaveGage;		//true if we want to use a wave gage
	float			xgage;
	float			ygage;
	//Rozita
	float3			gage[10];
	float			WaveGageNum;
	int				numbodies;			// number of floating bodies
	uint			maxneibsnum;		// maximum number of neibs (should be a multiple of NEIBS_INTERLEAVE)
	SimParams(void) :
		kernelradius(2.0),
		dt(0.00013),
		tend(0),
		xsph(false),
		dtadapt(true),
		dtadaptfactor(0.3),
		buildneibsfreq(10),
		shepardfreq(0),
		mlsfreq(15),
		visctype(ARTVISC),
		mbcallback(false),
		gcallback(false),
		periodicbound(false),
		nlexpansionfactor(1.0),
		usedem(false),
		sph_formulation(SPH_F1),
		boundarytype(LJ_BOUNDARY),
		vorticity(false),
		testpoints(false),
		savenormals(false),
		surfaceparticle(false),
		//WaveGage 
		writeWaveGage (false),
		xgage(0),
		ygage(0),
		numbodies(0),
		maxneibsnum(128)
	{};
} SimParams;

#endif

