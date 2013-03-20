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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#include "buildneibs.cuh" // hashKey
#include "Problem.h"
#include "Writer.h"

#include <exception>

class TimingException: public std::exception
{

public:
	float simTime, dt;

	TimingException(float _time = nan(""), float _dt = nan("")) :
		std::exception(), simTime(_time), dt(_dt) {}

	virtual const char *what() const throw() {
		return "timing error";
	}
};

class DtZeroException: public TimingException
{
public:
	DtZeroException(float _time = nan(""), float _dt = 0) :
		TimingException(_time, _dt) {}

	virtual const char *what() const throw() {
		return "timestep zeroed!";
	}
};


// ParticleSystem : class used to call CUDA kernels
class ParticleSystem
{
	public: // TODO Two "public" keywords in this class, probably this one should be "private:"
		enum ParticleArray
		{
			POSITION = 0,
			VELOCITY,
			INFO,
			VORTICITY,
			FORCE,
			FORCENORM,
			NEIBSLIST,
			HASH,
			PARTINDEX,
			CELLSTART,
			CELLEND,
			// Free surface detection (Debug)
			NORMALS,
			BOUNDELEMENT,
			GRADGAMMA,
			VERTICES,
			PRESSURE,
			INVALID_PARTICLE_ARRAY
		};

		enum
		{
			VM_NORMAL = 0,
			VM_VELOCITY,
			VM_PRESSURE,
			VM_DENSITY,
			VM_VORTICITY,
			VM_NOFLUID
		};

		ParticleSystem(Problem *problem);
		~ParticleSystem();

		void	allocate(uint numParticles);
		void	allocate_planes(uint numPlanes);
		void	setPhysParams(void);
		void	getPhysParams(void);
		void	printPhysParams(FILE *summary = NULL);
		void	printSimParams(FILE *summary = NULL);

		void	buildNeibList(bool timing);
		void	initializeGammaAndGradGamma(void);
		void	imposeDynamicBoundaryConditions(void);
		void	updateValuesAtBoundaryElements(void);

		TimingInfo	PredcorrTimeStep(bool);

		void*	getArray(ParticleArray, bool);
		void	setArray(ParticleArray);
		void	setPlanes();
		void	drawParts(bool, bool, bool, int);
		void	writeSummary(void);
		void	writeToFile(void);

		int		getNumParticles() const { return m_numParticles; }

		long	getIter(void) { return m_iter; }
		float	getTimestep(void) { return m_dt; }
		float	getTime(void) { return m_simTime; }

		// DEBUG
		void saveneibs();
		void savehash();
		void saveindex();
		void savesorted();
		void savecellstartend();
		void savegradgamma();
		void saveboundelem();
		void savepressure();
		void saveprobedata();
		void saveVelocity();
		void saveForces();
		void reducerbforces();

		//WaveGage
		void writeWaveGage ();

		// Free surface detection (Debug)
		void savenormals();

	public:
		Problem		*m_problem;				// pointer to problem object

		bool		m_neiblist_built;	// has the neib list ever been built?

		// Physicals and simulation parameters
		PhysParams	*m_physparams;
		SimParams	*m_simparams;
		float		m_influenceRadius;		// slength*kernelRadius
		float		m_nlInfluenceRadius;	// influence radius for neib list construction
		float		m_nlSqInfluenceRadius;	// square influence radius for neib list construction
		float		m_dtprev;				// DEBUG: dt at previous iter

		// Informations for timing
		TimingInfo	m_timingInfo;

		// Geometrical datas and problem definition
		uint		m_numParticles;			// total number of particles
		uint		m_numPlanes;			// total number of planes
		uint3		m_gridSize;				// number of domain cells in each direction
		uint		m_nGridCells;			// total number of domain cells
		uint		m_nSortingBits;			// number of significant bits for sorting (ceil(log2(m_nGridCells)))
		float3		m_worldOrigin;			// origin of simulation domain
		float3		m_worldSize;			// simulation domain size
		float3		m_cellSize;				// size of domain cell

		// Simultaion time datas
		float		m_dt;					// time step
		float		m_simTime;				// simulation time
		long		m_iter;					// iteration number

		// CPU arrays
		float4*		m_hPos;					// postions array
		float4*		m_hVel;					// velocity array
		float4*		m_hForces;				// forces array
		particleinfo*	m_hInfo;			// info array
		float3*		m_hVort;				// vorticity
		float*		m_hVisc;				// viscosity
		float4*		m_hNormals;				// normals at free surface
		float4*		m_hEnergy;				// total fluid(s) energy

		// CPU arrays for geometry
		float4*		m_hPlanes;
		float *		m_hPlanesDiv;

		//CPU arrays for Ferrand et al. boundary model
		float4*		m_hGradGamma;			// gradient of renormalization term gamma (x,y,z) and gamma itself (w)
		float4*		m_hBoundElement;		// normal coordinates (x,y,z) and surface (w) of boundary elements (triangles)
		vertexinfo*	m_hVertices;			// stores indexes of 3 vertex particles for every boundary element
		float*		m_hPressure;			// stores pressure, used only for vertex and boundary particles

		// CPU arrays used for debugging
		uint*		m_hNeibsList;
		hashKey*	m_hParticleHash;
		uint*		m_hCellStart;
		uint*		m_hCellEnd;
		uint*		m_hParticleIndex;
		float4*		m_hRbForces;
		float4*		m_hRbTorques;

		// GPU arrays
		float4*		m_dForces;				// forces array
		float4*		m_dXsph;				// mean velocity array
		float4*		m_dPos[2];				// position array
		float4*		m_dVel[2];				// velocity array
		particleinfo*	m_dInfo[2];				// particle info array
		float4*		m_dNormals;				// normal at free surface
		float3*		m_dVort;				// vorticity
		uint		m_numPartsFmax;				// number of particles divided by BLOCK_SIZE
		float*		m_dCfl;					// cfl for each block
		float*		m_dCflGamma;				// analogue of cfl due to gamma integration
		float*		m_dTempCfl;				// temporary storage for cfl computation
		float*		m_dCfl2;				// test
		float2*		m_dTau[3];				// SPS stress tensor
		
		float4*		m_dGradGamma[2];			// gradient of renormalization term gamma (x,y,z) and gamma itself (w)
		float4*		m_dBoundElement[2];			// normal coordinates (x,y,z) and surface (w) of boundary elements (triangles)
		vertexinfo*	m_dVertices[2];				// stores indexes of 3 vertex particles for every boundary element
		float*		m_dPressure[2];				// stores pressure for vertex and boundary particles

		// TODO: profile with float3
		uint		m_numBodiesParticles;	// Total number of particles belonging to rigid bodies
		float4*		m_dRbForces;			// Forces on particles belonging to rigid bodies
		float4*		m_dRbTorques;			// Torques on particles belonging to rigid bodies
		uint*		m_dRbNum;				// Key used in segmented scan
		uint*		m_hRbLastIndex;			// Indexes of last particles belonging to rigid bodies
		float3*		m_hRbTotalForce;		// Total force acting on each rigid body
		float3*		m_hRbTotalTorque;		// Total torque acting on each rigid body

		uint		m_mbDataSize;			// size (in bytes) of m_dMbData array
		float4*		m_dMbData;				// device side moving boundary data

		hashKey*	m_dParticleHash;		// hash table for sorting
		uint*		m_dParticleIndex;		// sorted particle indexes
		uint*		m_dInversedParticleIndex;	// inversed m_dParticle index array
		uint*		m_dCellStart;			// index of cell start in sorted order
		uint*		m_dCellEnd;			// index of cell end in sorted order
		uint*		m_dNeibsList;			// neib list with MAXNEIBSNUM neibs per particle
		uint*		m_dNewNumParticles;		// number of active particles found during neib list

		uint		m_currentPosRead;		// current index in m_dPos for position reading (0 or 1)
		uint		m_currentPosWrite;		// current index in m_dPos for writing (0 or 1)
		uint		m_currentVelRead;		// current index in m_dVel for velocity reading (0 or 1)
		uint		m_currentVelWrite;		// current index in m_dVel for writing (0 or 1)
		uint		m_currentInfoRead;		// current index in m_dInfo for info reading (0 or 1)
		uint		m_currentInfoWrite;		// current index in m_dInfo for writing (0 or 1)
		uint		m_currentBoundElementRead;	// current index in m_dBoundElement for normal coordinates (and surface) reading (0 or 1)
		uint		m_currentBoundElementWrite;	// current index in m_dBoundElement for writing (0 or 1)
		uint		m_currentGradGammaRead;		// current index in m_dGradGamma for gradient gamma (and gamma) reading (0 or 1)
		uint		m_currentGradGammaWrite;	// current index in m_dGradGamma for writing (0 or 1)
		uint		m_currentVerticesRead;		// current index in m_dVertices for vertices reading (0 or 1)
		uint		m_currentVerticesWrite;		// current index in m_dVertices for writing (0 or 1)
		uint		m_currentPressureRead;		// current index in m_dPressure for pressure reading (0 or 1)
		uint		m_currentPressureWrite;		// current index in m_dPressure for writing (0 or 1)
		
		// CUDA device properties
		cudaDeviceProp	m_device;

		// File writer
		Problem::WriterType m_writerType;
		Writer		*m_writer;
};
#endif
