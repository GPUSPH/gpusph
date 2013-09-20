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

#include <cmath>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <stdexcept>

#include <float.h>

#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "ParticleSystem.h"

#include "TextWriter.h"
#include "CustomTextWriter.h"
#include "VTKWriter.h"
#include "VTKLegacyWriter.h"
#include "Problem.h"
/* Include only the problem selected at compile time */
#include "problem_select.opt"

#include "cudautil.cuh"
#include "buildneibs.cuh"
#include "forces.cuh"
#include "euler.cuh"

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/functional.h>

static const char* ParticleArrayName[ParticleSystem::INVALID_PARTICLE_ARRAY+1] = {
	"Position",
	"Velocity",
	"Info",
	"Vorticity",
	"Force",
	"Force norm",
	"Neighbor list",
	"Hash",
	"Particle Index",
	"Cell Start",
	"Cell End",
	"(invalid)"
};

ParticleSystem::ParticleSystem(Problem *problem) :
	m_problem(problem),
	m_physparams(problem->get_physparams()),
	m_simparams(problem->get_simparams()),
	m_simTime(0.0),
	m_iter(0),
	m_currentPosRead(0),
	m_currentPosWrite(1),
	m_currentVelRead(0),
	m_currentVelWrite(1),
	m_currentInfoRead(0),
	m_currentInfoWrite(1)
{
	m_worldOrigin = problem->get_worldorigin();
	m_worldSize = problem->get_worldsize();
	m_cellSize = problem->get_cellsize();
	m_gridSize = problem->get_gridsize();
	m_writerType = problem->get_writertype();

	m_influenceRadius = m_simparams.kernelradius*m_simparams.slength;
	m_nlInfluenceRadius = m_influenceRadius*m_simparams.nlexpansionfactor;
	m_nlSqInfluenceRadius = m_nlInfluenceRadius*m_nlInfluenceRadius;

	m_nGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
	m_nSortingBits = ceil(log2((float) m_nGridCells)/4.0)*4;

	m_dt = m_simparams.dt;

	m_timingInfo.dt = m_dt;
	m_timingInfo.t = 0.0f;
	m_timingInfo.maxNeibs = 0;
	m_timingInfo.numInteractions = 0.0f;
	m_timingInfo.meanNumInteractions = 0;
	m_timingInfo.iterations = 0;
	m_timingInfo.timeNeibsList = 0.0f;
	m_timingInfo.meanTimeNeibsList = 0.0f;
	m_timingInfo.timeInteract = 0.0f;
	m_timingInfo.meanTimeInteract = 0.0f;
	m_timingInfo.timeEuler = 0.0f;
	m_timingInfo.meanTimeEuler = 0.0f;

	// CHecking number of moving boundaries
	if (m_problem->m_mbnumber > MAXMOVINGBOUND) {
		stringstream ss;
		ss << "Number of moving boundaries " << m_problem->m_mbnumber <<
			" > MAXMOVINGBOUND (" << MAXMOVINGBOUND << ")" << endl;
		throw runtime_error(ss.str());
		}

	// Computing size of moving bloudaries data
	m_mbDataSize = m_problem->m_mbnumber*sizeof(float4);

	printf("GPU implementation\n");
	printf("Number of grid cells : %d\n", m_nGridCells);
	printf("Grid size : (%d, %d, %d)\n", m_gridSize.x, m_gridSize.y, m_gridSize.z);
	printf("Cell size : (%f, %f, %f)\n", m_cellSize.x, m_cellSize.y, m_cellSize.z);

	// CUDA init
	checkCUDA(problem->get_options());
	printf("\nCuda initialized\n");

	//setPhysParams();

	switch(m_writerType) {
		case Problem::TEXTWRITER:
			m_writer = new TextWriter(problem);
			break;

		case Problem::VTKWRITER:
			m_writer = new VTKWriter(problem);
			break;

		case Problem::VTKLEGACYWRITER:
			m_writer = new VTKLegacyWriter(problem);
			break;

		case Problem::CUSTOMTEXTWRITER:
			m_writer = new CustomTextWriter(problem);
			break;

		default:
			stringstream ss;
			ss << "Writer not supported";
			throw runtime_error(ss.str());
			break;
	}

}


void
ParticleSystem::allocate(uint numParticles)
{
	m_numParticles = numParticles;
	m_timingInfo.numParticles = numParticles;

	setPhysParams();
	writeSummary();

	// allocate host storage
	const uint memSize = sizeof(float)*m_numParticles;
	const uint memSize2 = sizeof(float2)*m_numParticles;
	const uint memSize3 = sizeof(float3)*m_numParticles;
	const uint memSize4 = sizeof(float4)*m_numParticles;
	const uint infoSize = sizeof(particleinfo)*m_numParticles;
	const uint hashSize = sizeof(uint)*m_numParticles;
	const uint gridcellSize = sizeof(uint)*m_nGridCells;
	const uint neibslistSize = sizeof(neibdata)*m_simparams.maxneibsnum*m_numParticles;

	uint memory = 0;

	m_hPos = new float4[m_numParticles];
	memset(m_hPos, 0, memSize4);
	memory += memSize4;
	m_hdPos = new double4[m_numParticles];
	memset(m_hdPos, 0, 2*memSize4);
	memory += 2*memSize4;

	m_hVel = new float4[m_numParticles];
	memset(m_hVel, 0, memSize4);
	memory += memSize4;

	m_hInfo = new particleinfo[m_numParticles];
	memset(m_hInfo, 0, infoSize);
	memory += infoSize;

	m_hParticleHash = new uint[m_numParticles];
	memset(m_hParticleHash, 0, hashSize);
	memory += hashSize;

	m_hVort = NULL;
	if (m_simparams.vorticity) {
		m_hVort = new float3[m_numParticles];
		memory += memSize3;
		}


#ifdef PSYSTEM_DEBUG
	m_hForces = new float4[m_numParticles];
	memset(m_hForces, 0, memSize4);
	memory += memSize4;

	m_hParticleIndex = new uint[m_numParticles];
	memset(m_hParticleIndex, 0, hashSize);
	memory += hashSize;

	m_hCellStart = new uint[m_nGridCells];
	memset(m_hCellStart, 0, gridcellSize);
	memory += gridcellSize;

	m_hCellEnd = new uint[m_nGridCells];
	memset(m_hCellEnd, 0, gridcellSize);
	memory += gridcellSize;

	m_hNeibsList = new neibdata[m_simparams.maxneibsnum*(m_numParticles/NEIBINDEX_INTERLEAVE + 1)*NEIBINDEX_INTERLEAVE];
	memset(m_hNeibsList, 0xffff, neibslistSize);
	memory += neibslistSize;
#endif

	// Free surface detection
	m_hNormals = NULL;
	if (m_simparams.savenormals) {
		m_hNormals = new float4[m_numParticles];
		memset(m_hNormals, 0, memSize4);
		memory += memSize4;
	}

	printf("\nCPU memory allocated\n");
	printf("Number of particles : %d\n", m_numParticles);
	printf("CPU memory used : %.2f MB\n", memory/(1024.0*1024.0));
	fflush(stdout);

	// allocate GPU data
	memory = 0;
	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dForces, memSize4));
	memory += memSize4;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dXsph, memSize4));
	memory += memSize4;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dPos[0], memSize4));
	memory += memSize4;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dPos[1], memSize4));
	memory += memSize4;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dVel[0], memSize4));
	memory += memSize4;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dVel[1], memSize4));
	memory += memSize4;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dInfo[0], infoSize));
	memory += infoSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dInfo[1], infoSize));
	memory += infoSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dParticleHash, hashSize));
	memory += hashSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dParticleIndex, hashSize));
	memory += hashSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dCellStart, gridcellSize));
	memory += gridcellSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dCellEnd, gridcellSize));
	memory += gridcellSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dNeibsList, neibslistSize));
	memory += neibslistSize;

	// Free surface detection
	if (m_simparams.savenormals) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dNormals, memSize4));
		memory += memSize4;
	}

	if (m_simparams.vorticity) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dVort, memSize3));
		memory += memSize3;
	}

	if (m_simparams.visctype == SPSVISC) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTau[0], memSize2));
		memory += memSize2;

		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTau[1], memSize2));
		memory += memSize2;

		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTau[2], memSize2));
		memory += memSize2;
	}

	// Allocate storage for rigid bodies forces and torque computation
	if (m_simparams.numODEbodies) {
		m_numBodiesParticles = m_problem->get_ODE_bodies_numparts();
		printf("number of rigid bodies particles = %d\n", m_numBodiesParticles);
		int memSizeRbForces = m_numBodiesParticles*sizeof(float4);
		int memSizeRbNum = m_numBodiesParticles*sizeof(uint);
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dRbTorques, memSizeRbForces));
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dRbForces, memSizeRbForces));
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dRbNum, memSizeRbNum));
		memory += 2*memSizeRbForces + memSizeRbNum;

		// DEBUG
		m_hRbForces = new float4[m_numBodiesParticles];
		m_hRbTorques = new float4[m_numBodiesParticles];

		uint rbfirstindex[MAXBODIES];
		uint* rbnum = new uint[m_numBodiesParticles];
		m_hRbLastIndex = new uint[m_simparams.numODEbodies];
		m_hRbTotalForce = new float3[m_simparams.numODEbodies];
		m_hRbTotalTorque = new float3[m_simparams.numODEbodies];

		rbfirstindex[0] = 0;
		for (int i = 1; i < m_simparams.numODEbodies; i++) {
			rbfirstindex[i] = rbfirstindex[i - 1] + m_problem->get_ODE_body_numparts(i - 1);
		}
		setforcesrbstart(rbfirstindex, m_simparams.numODEbodies);

		int offset = 0;
		for (int i = 0; i < m_simparams.numODEbodies; i++) {
			m_hRbLastIndex[i] = m_problem->get_ODE_body_numparts(i) - 1 + offset;
			for (int j = 0; j < m_problem->get_ODE_body_numparts(i); j++) {
				rbnum[offset + j] = i;
			}
			offset += m_problem->get_ODE_body_numparts(i);
		}
		size_t  size = m_numBodiesParticles*sizeof(uint);
		CUDA_SAFE_CALL(cudaMemcpy((void *) m_dRbNum, (void*) rbnum, size, cudaMemcpyHostToDevice));

		delete[] rbnum;
	}

	if (m_simparams.dtadapt) {
		m_numPartsFmax = getNumPartsFmax(numParticles);
		const uint fmaxTableSize = m_numPartsFmax*sizeof(float);

		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dCfl, fmaxTableSize));
		CUDA_SAFE_CALL(cudaMemset(m_dCfl, 0, fmaxTableSize));

		const uint tempCflSize = getFmaxTempStorageSize(m_numPartsFmax);
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTempCfl, tempCflSize));
		CUDA_SAFE_CALL(cudaMemset(m_dTempCfl, 0, tempCflSize));

		memory += fmaxTableSize;
		}

	// Allocating, reading and copying DEM
	if (m_simparams.usedem) {
		printf("Using DEM\n");
		printf("cols = %d\trows =% d\n", m_problem->m_ncols, m_problem->m_nrows);
		setDemTexture(m_problem->m_dem, m_problem->m_ncols, m_problem->m_nrows);
		}
	printf("Number of fluids: %d\n",m_physparams.numFluids);
	printf("GPU memory allocated\n");
	printf("GPU memory used : %.2f MB\n", memory/(1024.0*1024.0));

	fflush(stdout);
}


void
ParticleSystem::allocate_planes(uint numPlanes)
{
	m_numPlanes = numPlanes;

	const uint planeSize4 = sizeof(float4)*m_numPlanes;
	const uint planeSize  = sizeof(float )*m_numPlanes;

	uint memory = 0;

	m_hPlanes = new float4[m_numPlanes];
	memset(m_hPlanes, 0, planeSize4);
	memory += planeSize4;

	m_hPlanesDiv = new float[m_numPlanes];
	memset(m_hPlanesDiv, 0, planeSize);
	memory += planeSize;

	printf("Number of planes : %d\n", m_numPlanes);
	printf("CPU memory used : %.2f MB\n", memory/(1024.0*1024.0));
	fflush(stdout);
}


void
ParticleSystem::setPhysParams(void)
{
	// Setting visccoeff
	switch (m_simparams.visctype) {
		case ARTVISC:
			m_physparams.visccoeff = m_physparams.artvisccoeff;
			break;

		case KINEMATICVISC:
		case SPSVISC:
			m_physparams.visccoeff = 4.0*m_physparams.kinematicvisc;
			break;

		case DYNAMICVISC:
			m_physparams.visccoeff = m_physparams.kinematicvisc;
			break;
	}

	setforcesconstants(m_simparams, m_physparams, m_gridSize, m_cellSize, m_numParticles);
	seteulerconstants(m_physparams, m_gridSize, m_cellSize);
	setneibsconstants(m_simparams, m_physparams);
}


void
ParticleSystem::getPhysParams(void)
{
	getforcesconstants(m_physparams);
	geteulerconstants(m_physparams);
	getneibsconstants(m_simparams, m_physparams);
}


void
ParticleSystem::printPhysParams(FILE *summary)
{
	if (!summary)
		summary = stdout;
    getPhysParams();
	fprintf(summary, "\nPhysical parameters:\n");

	unsigned int i=0;

	while(i < m_physparams.numFluids) {
		fprintf(summary,"fluid #%u \n", i);
		fprintf(summary, "\trho0 = %g\n", m_physparams.rho0[i]);
		fprintf(summary, "\tb = %g\n", m_physparams.bcoeff[i]);
		fprintf(summary, "\tgamma = %g\n", m_physparams.gammacoeff[i]);
		fprintf(summary, "\tsscoeff = %g\n", m_physparams.sscoeff[i]);
		fprintf(summary, "\tsspowercoeff = %g\n", m_physparams.sspowercoeff[i]);
		fprintf(summary, "\tsound speed = %g\n", m_problem->soundspeed(m_physparams.rho0[i],i));
		++i;
		}

#define g m_physparams.gravity
	fprintf(summary, "gravity = (%g, %g, %g) [%g]\n", g.x, g.y, g.z, length(g));
#undef g

	fprintf(summary, "%s boundary parameters:\n", BoundaryName[m_simparams.boundarytype]);
	switch (m_simparams.boundarytype) {
		case LJ_BOUNDARY:
			fprintf(summary, "\td = %g\n", m_physparams.dcoeff);
			fprintf(summary, "\tp1 = %g\n", m_physparams.p1coeff);
			fprintf(summary, "\tp2 = %g\n", m_physparams.p2coeff);
			break;
		case MK_BOUNDARY:
			fprintf(summary, "\tK = %g\n", m_physparams.MK_K);
			fprintf(summary, "\td = %g\n", m_physparams.MK_d);
			fprintf(summary, "\tbeta = %g\n", m_physparams.MK_beta);
			break;
	}

	fprintf(summary, "r0 = %g\n", m_physparams.r0);
	float visccoeff = m_physparams.visccoeff;
	fprintf(summary,"Viscosity\n");
	switch (m_simparams.visctype) {
		case ARTVISC:
			fprintf(summary, "\tArtificial viscosity: artvisccoeff = %g (viscoeff=%g = artvisccoeff)\n",
					m_physparams.artvisccoeff, visccoeff);
			break;

		case KINEMATICVISC:
			fprintf(summary, "\tKinematic viscosity: kinematicvisc = %g m^2/s (viscoeff=%g = 4*kinematicvisc)\n",
					m_physparams.artvisccoeff, visccoeff);
			break;

		case DYNAMICVISC:
			fprintf(summary, "\tDynamic viscosity: kinematicvisc = %g m^2/s (viscoeff=%g = kinematicvisc)\n",
					m_physparams.artvisccoeff, visccoeff);
			break;

		case SPSVISC:
			fprintf(summary, "\tSPS + kinematic viscosity: kinematicvisc = %g m^2/s (viscoeff=%g = kinematicvisc)\n",
					m_physparams.artvisccoeff, visccoeff);
			fprintf(summary, "\tSmagFactor = %g\n", m_physparams.smagfactor);
			fprintf(summary, "\tkSPSFactor = %g\n", m_physparams.kspsfactor);
			break;
		}
	fprintf(summary, "espartvisc = %g\n", m_physparams.epsartvisc);
	fprintf(summary, "epsxsph = %g\n", m_physparams.epsxsph);
	if (m_simparams.periodicbound) {
		fprintf(summary, "Periodic boundary parameters in x %d, y %d, z %d\n", m_simparams.periodicbound & XPERIODIC,
				m_simparams.periodicbound & YPERIODIC, m_simparams.periodicbound & ZPERIODIC);
		}
	if (m_simparams.usedem) {
		fprintf(summary, "DEM resolution ew = %g, ns = %g\n", m_physparams.ewres, m_physparams.nsres);
		fprintf(summary, "Displacement for normal computing dx = %g, dy = %g\n", m_physparams.demdx, m_physparams.demdy);
		fprintf(summary, "DEM zmin = %g\n", m_physparams.demzmin);
		}

}


void
ParticleSystem::printSimParams(FILE *summary)
{
	if (!summary)
		summary = stdout;

	fprintf(summary, "\nSimulation parameters:\n");
	fprintf(summary, "slength = %g\n", m_simparams.slength);
	fprintf(summary, "kernelradius = %g\n", m_simparams.kernelradius);
	fprintf(summary, "initial dt = %g\n", m_simparams.dt);
	fprintf(summary, "simulation end time = %g\n", m_simparams.tend);
	fprintf(summary, "maximum number of neighbors per particle = %d\n", m_simparams.maxneibsnum);
	fprintf(summary, "neib list construction every %d iteration\n", m_simparams.buildneibsfreq);
	fprintf(summary, "Shepard filter every %d iteration\n", m_simparams.shepardfreq);
	fprintf(summary, "MLS filter every %d iteration\n", m_simparams.mlsfreq);
	fprintf(summary, "adaptive time step = %d\n", m_simparams.dtadapt);
	fprintf(summary, "safety factor for adaptive time step = %f\n", m_simparams.dtadaptfactor);
	fprintf(summary, "xsph correction = %d\n", m_simparams.xsph);
	fprintf(summary, "SPH formulation = %d\n", m_simparams.sph_formulation);
	fprintf(summary, "viscosity type = %d (%s)\n", m_simparams.visctype, ViscosityName[m_simparams.visctype]);
	fprintf(summary, "moving boundary velocity callback function = %d (0 none)\n", m_simparams.mbcallback);
	if (m_simparams.mbcallback)
		fprintf(summary, "\tnumber of moving boundaries = %d\n", m_problem->m_mbnumber);
	fprintf(summary, "variable gravity callback function = %d\n",m_simparams.gcallback);
	fprintf(summary, "periodic boundary = %s\n", m_simparams.periodicbound ? "true" : "false");
	fprintf(summary, "using DEM = %d\n", m_simparams.usedem);
	fprintf(summary, "number of rigid bodies = %d\n", m_simparams.numODEbodies);
}


void
ParticleSystem::writeSummary(void)
{

	std::string fname;
	fname = m_problem->get_dirname() + "/summary.txt";
	FILE *summary = fopen(fname.c_str(),"w");

	printSimParams(summary);
	printPhysParams(summary);

	fclose(summary);
}

// TODO: restore DEBUG define and do some cleaning on TestParticle and SaveNormals
ParticleSystem::~ParticleSystem()
{
	delete [] m_hPos;
	delete [] m_hdPos;
	delete [] m_hVel;
	delete [] m_hInfo;
	if (m_simparams.vorticity) {
		delete [] m_hVort;
		}

#ifdef PSYSTEM_DEBUG
	delete [] m_hForces;
	delete [] m_hNeibsList;
	delete [] m_hParticleHash;
	delete [] m_hParticleIndex;
	delete [] m_hCellStart;
	delete [] m_hCellEnd;
#endif
	if (m_simparams.savenormals)
		delete [] m_hNormals;

	delete m_writer;

	CUDA_SAFE_CALL(cudaFree(m_dForces));
	CUDA_SAFE_CALL(cudaFree(m_dXsph));

	if (m_simparams.visctype == SPSVISC) {
		CUDA_SAFE_CALL(cudaFree(m_dTau[0]));
		CUDA_SAFE_CALL(cudaFree(m_dTau[1]));
		CUDA_SAFE_CALL(cudaFree(m_dTau[2]));
		}

	if (m_simparams.vorticity) {
		CUDA_SAFE_CALL(cudaFree(m_dVort));
		}

	CUDA_SAFE_CALL(cudaFree(m_dPos[0]));
	CUDA_SAFE_CALL(cudaFree(m_dPos[1]));
	CUDA_SAFE_CALL(cudaFree(m_dVel[0]));
	CUDA_SAFE_CALL(cudaFree(m_dVel[1]));
	CUDA_SAFE_CALL(cudaFree(m_dInfo[0]));
	CUDA_SAFE_CALL(cudaFree(m_dInfo[1]));

	// Free surface detection
	if (m_simparams.savenormals)
		CUDA_SAFE_CALL(cudaFree(m_dNormals));

	if (m_simparams.numODEbodies) {
		delete [] m_hRbLastIndex;
		delete [] m_hRbTotalForce;
		delete [] m_hRbTotalTorque;
		CUDA_SAFE_CALL(cudaFree(m_dRbTorques));
		CUDA_SAFE_CALL(cudaFree(m_dRbForces));
		CUDA_SAFE_CALL(cudaFree(m_dRbNum));

		// DEBUG
		delete [] m_hRbForces;
		delete [] m_hRbTorques;
		}

	CUDA_SAFE_CALL(cudaFree(m_dParticleHash));
	CUDA_SAFE_CALL(cudaFree(m_dParticleIndex));
	CUDA_SAFE_CALL(cudaFree(m_dCellStart));
	CUDA_SAFE_CALL(cudaFree(m_dCellEnd));
	CUDA_SAFE_CALL(cudaFree(m_dNeibsList));

	if (m_simparams.usedem)
		releaseDemTexture();

	if (m_simparams.dtadapt) {
		CUDA_SAFE_CALL(cudaFree(m_dCfl));
		CUDA_SAFE_CALL(cudaFree(m_dTempCfl));
		}

	printf("GPU and CPU memory released\n\n");
}


// TODO: DEBUG, testpoints, freesurface and size_t
void
ParticleSystem::getArray(ParticleArray array)
{
	void*   hdata = 0;
	void*   ddata = 0;
	long	size = 0;;

	switch (array) {
		default:
		case POSITION:
			size = m_numParticles*sizeof(float4);
			hdata = (void*) m_hPos;
			ddata = (void*) m_dPos[m_currentPosRead];
			break;

		case VELOCITY:
			if (m_simparams.testpoints) {
				testpoints(	m_dPos[m_currentPosRead],
							m_dVel[m_currentVelRead],
							m_dInfo[m_currentInfoRead],
							m_dParticleHash,
							m_dCellStart,
							m_dNeibsList,
							m_numParticles,
							m_simparams.slength,
							m_simparams.kerneltype,
							m_influenceRadius);
				}

			size = m_numParticles*sizeof(float4);
			hdata = (void*) m_hVel;
			ddata = (void*) m_dVel[m_currentVelRead];
			break;

		case INFO:
			if (m_simparams.surfaceparticle) {
				surfaceparticle( m_dPos[m_currentPosRead],
								 m_dVel[m_currentVelRead],
								 m_dNormals,
								 m_dInfo[m_currentInfoRead],
								 m_dInfo[m_currentInfoWrite],
								 m_dParticleHash,
								 m_dCellStart,
								 m_dNeibsList,
								 m_numParticles,
								 m_simparams.slength,
								 m_simparams.kerneltype,
								 m_influenceRadius,
								 m_simparams.savenormals);
				std::swap(m_currentInfoRead, m_currentInfoWrite);
				if (m_simparams.savenormals) {
					size = m_numParticles*sizeof(float4);
					CUDA_SAFE_CALL(cudaMemcpy((void*) m_hNormals, (void*) m_dNormals, size, cudaMemcpyDeviceToHost));
					}
				}

			size = m_numParticles*sizeof(particleinfo);
			hdata = (void*) m_hInfo;
			ddata = (void*) m_dInfo[m_currentInfoRead];
			break;

		case VORTICITY:
			if (m_simparams.vorticity) {
				size = m_numParticles*sizeof(float3);
				hdata = (void*) m_hVort;
				ddata = (void*) m_dVort;

				// Calling vorticity computation kernel
				vorticity(	m_dPos[m_currentPosRead],
							m_dVel[m_currentVelRead],
							m_dVort,
							m_dInfo[m_currentInfoRead],
							m_dParticleHash,
							m_dCellStart,
							m_dNeibsList,
							m_numParticles,
							m_simparams.slength,
							m_simparams.kerneltype,
							m_influenceRadius);
				}
			else
				size = 0;
			break;

		case HASH:
			size = m_numParticles*sizeof(uint);
			hdata = (void*) m_hParticleHash;
			ddata = (void*) m_dParticleHash;
			break;

#ifdef _DEBUG_
		case FORCE:
			size = m_numParticles*sizeof(float4);
			hdata = (void*) m_hForces;
			ddata = (void*) m_dForces;
			break;

		case NEIBSLIST:
			size = m_numParticles*m_simparams.maxneibsnum*sizeof(neibdata);
			hdata = (void*) m_hNeibsList;
			ddata = (void*) m_dNeibsList;
			break;

		case PARTINDEX:
			size = m_numParticles*sizeof(uint);
			hdata = (void*) m_hParticleIndex;
			ddata = (void*) m_dParticleIndex;
			break;

		case CELLSTART:
			size = m_numParticles*sizeof(uint);
			hdata = (void*) m_hCellStart;
			ddata = (void*) m_dCellStart;
			break;

		case CELLEND:
			size = m_numParticles*sizeof(uint);
			hdata = (void*) m_hCellEnd;
			ddata = (void*) m_dCellEnd;
			break;
#endif
	}

	if (size)
		CUDA_SAFE_CALL(cudaMemcpy(hdata, ddata, size, cudaMemcpyDeviceToHost));
}


// TODO: size_t, char *
void
ParticleSystem::setArray(ParticleArray array)
{
	void* hdata = 0;
	void* ddata = 0;
	long  size;

	switch (array) {
		case POSITION:
			hdata = (void*) m_hPos;
			ddata = (void*) m_dPos[m_currentPosRead];
			size = m_numParticles*sizeof(float4);
			break;

		case VELOCITY:
			hdata = (void*) m_hVel;
			ddata = (void*) m_dVel[m_currentVelRead];
			size = m_numParticles*sizeof(float4);
			break;

		case INFO:
			hdata = (void*) m_hInfo;
			ddata = (void*) m_dInfo[m_currentInfoRead];
			size = m_numParticles*sizeof(particleinfo);
			break;

		case HASH:
			hdata = (void*) m_hParticleHash;
			ddata = (void*) m_dParticleHash;
			size = m_numParticles*sizeof(uint);
			break;

		default:
			fprintf(stderr, "Trying to upload unknown array %d\n", array);
			return;
	}

	printf("Uploading array %d (%s)\n", array, ParticleArrayName[array]);
	CUDA_SAFE_CALL(cudaMemcpy((char *) ddata, hdata, size, cudaMemcpyHostToDevice));
}


void
ParticleSystem::setPlanes(void)
{
	printf("Uploading %u planes\n", m_numPlanes);
	setplaneconstants(m_numPlanes, m_hPlanesDiv, m_hPlanes);
}


// TODO: check writer for testpoints
void
ParticleSystem::writeToFile()
{
	for (uint i = 0; i < m_numParticles; i++) {
		const float4 pos = m_hPos[i];
		double4 dpos;
		uint3 gridPos = calcGridPos(m_hParticleHash[i]);
		dpos.x = ((double) m_cellSize.x)*(gridPos.x + 0.5) + (double) pos.x;
		dpos.y = ((double) m_cellSize.y)*(gridPos.y + 0.5) + (double) pos.y;
		dpos.z = ((double) m_cellSize.z)*(gridPos.z + 0.5) + (double) pos.z;

		m_hdPos[i] = dpos;
	}
	//Testpoints
	m_writer->write(m_numParticles, m_hdPos, m_hVel, m_hInfo, m_hVort, m_simTime, m_simparams.testpoints, m_hNormals);
}


uint3
ParticleSystem::calcGridPos(uint particleHash)
{
	uint3 gridPos;
	gridPos.z = particleHash/(m_gridSize.x*m_gridSize.y);
	gridPos.y = (particleHash - gridPos.z*m_gridSize.x*m_gridSize.y)/m_gridSize.x;
	gridPos.x = particleHash - gridPos.y*m_gridSize.x - gridPos.z*m_gridSize.x*m_gridSize.y;

	return gridPos;
}

void
ParticleSystem::drawParts(bool show_boundary, bool show_floating, int view_mode)
{
	float minrho = m_problem->get_minrho();
	float maxrho = m_problem->get_maxrho();
	float minvel = m_problem->get_minvel();
	float maxvel = m_problem->get_maxvel();

	float minvort = 0.0;
	float maxvort = 2.0;


	float minp = m_problem->pressure(minrho,0); //FIX FOR MULT-FLUID
	float maxp = m_problem->pressure(maxrho,0);

	float4* relpos = m_hPos;
	float4* vel = m_hVel;
	float3* vort = m_hVort;
	uint* 	hash = m_hParticleHash;
	particleinfo* info = m_hInfo;

	glPointSize(2.0);
	glBegin(GL_POINTS);
	{
		for (uint i = 0; i < m_numParticles; i++) {
			float3 pos = make_float3(relpos[i]);
			pos = m_cellSize*calcGridPos(hash[i]) + pos + 0.5f*m_cellSize;

			if (NOT_FLUID(info[i]) && !OBJECT(info[i]) && show_boundary) {
				glColor3f(0.0, 1.0, 0.0);
				glVertex3fv((float*)&pos);
			}
			if (OBJECT(info[i]) && show_floating) {
				glColor3f(1.0, 0.0, 0.0);
				glVertex3fv((float*)&pos);
			}
			if (FLUID(info[i])) {
				float v; unsigned int t;
				float ssvel = m_problem->soundspeed(vel[i].w, PART_FLUID_NUM(info[i]));
				switch (view_mode) {
					case VM_NORMAL:
					    glColor3f(0.0,0.0,1.0);
					    if (m_physparams.numFluids > 1) {
					       v = (float) PART_FLUID_NUM(info[i]);
	                       v /= (m_physparams.numFluids - 1);
						   glColor3f(v, 0.0, 1.0 - v);
						   }
						break;

					case VM_VELOCITY:
						v = length(make_float3(vel[i]));

						if (v > ssvel)
							printf("WARNING [%g]: particle %d speed %g > %g\n",
							m_simTime, i, v, ssvel);
						if (v*m_dtprev > m_influenceRadius)
							printf("WARNING [%g]: particle %d moved by %g > %g\n",
											m_simTime, i, v*m_dtprev, m_influenceRadius);
						glColor3f((v - minvel)/(maxvel - minvel), 0.0, 1 - (v - minvel)/(maxvel - minvel));
						break;

					case VM_DENSITY:
						v = vel[i].w;
						glColor3f((v - minrho)/(maxrho - minrho), 0.0,
								1 - (v - minrho)/(maxrho - minrho));
						break;

					case VM_PRESSURE:
						v = m_problem->pressure(vel[i].w, PART_FLUID_NUM(info[i]));
						glColor3f((v - minp)/(maxp - minp),
								1 - (v - minp)/(maxp - minp),0.0);
						break;

					case VM_VORTICITY:
						if (m_simparams.vorticity) {
							v = length(vort[i]);
							glColor3f(1.-(v-minvort)/(maxvort-minvort),1.0,1.0);
						}
						else
							glColor3f(1.0, 1.0, 0.0);

					    break;
				}
				glVertex3fv((float*)&pos);
			}
		}

	}
	glEnd();
}


TimingInfo
ParticleSystem::PredcorrTimeStep(bool timing)
{
	// do nothing if the simulation is over
	if (m_problem->finished(m_simTime))
		return m_timingInfo;

	cudaEvent_t start_neibslist, stop_neibslist;
	cudaEvent_t start_interactions, stop_interactions;
	cudaEvent_t start_euler, stop_euler;

	if (m_iter % m_simparams.buildneibsfreq == 0) {

		uint3 gridSize = m_gridSize;
		float3 cellSize = m_cellSize;
		float3 worldOrigin = m_worldOrigin;

		if (timing) {
			cudaEventCreate(&start_neibslist);
			cudaEventCreate(&stop_neibslist);
			cudaEventRecord(start_neibslist, 0);
			}

		// compute hash
		calcHash(m_dPos[m_currentPosRead],
				 m_dParticleHash,
				 m_dParticleIndex,
				 m_dInfo[m_currentInfoRead],
				 gridSize,
				 cellSize,
				 worldOrigin,
				 m_numParticles,
				 m_simparams.periodicbound);

		// hash based particle sort
		sort(m_dParticleHash, m_dParticleIndex, m_numParticles);

		reorderDataAndFindCellStart(m_dCellStart,	  // output: cell start index
									m_dCellEnd,		// output: cell end index
									m_dPos[m_currentPosWrite],		 // output: sorted positions
									m_dVel[m_currentVelWrite],		 // output: sorted velocities
									m_dInfo[m_currentInfoWrite],		 // output: sorted info
									m_dParticleHash,   // input: sorted grid hashes
									m_dParticleIndex,  // input: sorted particle indices
									m_dPos[m_currentPosRead],		 // input: unsorted positions
									m_dVel[m_currentVelRead],		 // input: unsorted velocities
									m_dInfo[m_currentInfoRead],		 // input: unsorted info
									m_numParticles,
									m_nGridCells);

		std::swap(m_currentPosRead, m_currentPosWrite);
		std::swap(m_currentVelRead, m_currentVelWrite);
		std::swap(m_currentInfoRead, m_currentInfoWrite);

		m_timingInfo.numInteractions = 0;
		m_timingInfo.maxNeibs = 0;
		resetneibsinfo();

		// Build the neibghors list
		buildNeibsList(	m_dNeibsList,
						m_dPos[m_currentPosRead],
						m_dInfo[m_currentInfoRead],
						m_dParticleHash,
						m_dCellStart,
						m_dCellEnd,
						gridSize,
						cellSize,
						worldOrigin,
						m_numParticles,
						m_nGridCells,
						m_nlSqInfluenceRadius,
						m_simparams.periodicbound);

		getneibsinfo(m_timingInfo);

		if (m_timingInfo.maxNeibs > m_simparams.maxneibsnum) {
			printf("WARNING: current max. neighbors numbers %d greater than MAXNEIBSNUM (%d)\n", m_timingInfo.maxNeibs, m_simparams.maxneibsnum);
			fflush(stdout);
			}

		if (timing) {
			cudaEventRecord(stop_neibslist, 0);
			cudaEventSynchronize(stop_neibslist);
			cudaEventElapsedTime(&m_timingInfo.timeNeibsList, start_neibslist, stop_neibslist);
			m_timingInfo.timeNeibsList *= 1e-3;
			cudaEventDestroy(start_neibslist);
			cudaEventDestroy(stop_neibslist);

			int iter = m_iter/m_simparams.buildneibsfreq + 1;
			m_timingInfo.meanNumInteractions = (m_timingInfo.meanNumInteractions*(iter - 1) + m_timingInfo.numInteractions)/iter;
			m_timingInfo.meanTimeNeibsList = (m_timingInfo.meanTimeNeibsList*(iter - 1) + m_timingInfo.timeNeibsList)/iter;
			}
	}


	if (m_simparams.shepardfreq > 0 && m_iter > 0 && (m_iter % m_simparams.shepardfreq == 0)) {
		shepard(m_dPos[m_currentPosRead],
				m_dVel[m_currentVelRead],
				m_dVel[m_currentVelWrite],
				m_dInfo[m_currentInfoRead],
				m_dParticleHash,
				m_dCellStart,
				m_dNeibsList,
				m_numParticles,
				m_simparams.slength,
				m_simparams.kerneltype,
				m_influenceRadius);

		std::swap(m_currentVelRead, m_currentVelWrite);
	}

	if (m_simparams.mlsfreq > 0 && m_iter > 0 && (m_iter % m_simparams.mlsfreq == 0)) {
		mls(	m_dPos[m_currentPosRead],
				m_dVel[m_currentVelRead],
				m_dVel[m_currentVelWrite],
				m_dInfo[m_currentInfoRead],
				m_dParticleHash,
				m_dCellStart,
				m_dNeibsList,
				m_numParticles,
				m_simparams.slength,
				m_simparams.kerneltype,
				m_influenceRadius);

		std::swap(m_currentVelRead, m_currentVelWrite);
	}


	m_dtprev = m_dt;
	float dt1 = 0.0f, dt2 = 0.0f;

	// Timing
	if (timing) {
		cudaEventCreate(&start_interactions);
		cudaEventCreate(&stop_interactions);
		cudaEventRecord(start_interactions, 0);
		}

	// setting moving boundaries data if necessary
	if (m_simparams.mbcallback) {
		float4* hMbData = m_problem->get_mbdata(m_simTime + m_dt/2.0, m_dt/2.0, m_iter == 0);
		if (hMbData)
			setmbdata(hMbData, m_mbDataSize);
		}
	if (m_simparams.gcallback) {
		m_physparams.gravity = m_problem->g_callback(m_simTime);
		setgravity(m_physparams.gravity);
	}

	float3 *cg;
	float3 *trans;
	float *rot;

	// Copying floating bodies centers of gravity for torque computation in forces (needed only at first
	// step)
	if (m_simparams.numODEbodies && m_iter == 0) {
		cg = m_problem->get_ODE_bodies_cg();
		setforcesrbcg(cg, m_simparams.numODEbodies);
		seteulerrbcg(cg, m_simparams.numODEbodies);
		}

	dt1 = forces(   m_dPos[m_currentPosRead],   // pos(n)
					m_dVel[m_currentVelRead],   // vel(n)
					m_dForces,					// f(n)
					m_dRbForces,
					m_dRbTorques,
					m_dXsph,
					m_dInfo[m_currentInfoRead],
					m_dParticleHash,
					m_dCellStart,
					m_dNeibsList,
					m_numParticles,
					m_simparams.slength,
					m_dt,
					m_simparams.dtadapt,
					m_simparams.dtadaptfactor,
					m_simparams.xsph,
					m_simparams.kerneltype,
					m_influenceRadius,
					m_simparams.visctype,
					m_physparams.visccoeff,
					m_dCfl,
					m_dTempCfl,
					m_numPartsFmax,
					m_dTau,
					m_simparams.sph_formulation,
					m_simparams.boundarytype,
					m_simparams.usedem);
	// At this point forces = f(pos(n), vel(n))

	if (timing) {
		cudaEventRecord(stop_interactions, 0);
		cudaEventSynchronize(stop_interactions);
		cudaEventElapsedTime(&m_timingInfo.timeInteract, start_interactions, stop_interactions);
		m_timingInfo.timeInteract *= 1e-3;
		cudaEventDestroy(start_interactions);
		cudaEventDestroy(stop_interactions);

		cudaEventCreate(&start_euler);
		cudaEventCreate(&stop_euler);
		cudaEventRecord(start_euler, 0);
		}

	euler(  m_dPos[m_currentPosRead],   // pos(n)
			m_dParticleHash,
			m_dVel[m_currentVelRead],   // vel(n)
			m_dInfo[m_currentInfoRead], //particleInfo(n)
			m_dForces,					// f(n)
			m_dXsph,
			m_dPos[m_currentPosWrite],  // pos(n+1/2) = pos(n) + vel(n)*dt/2
			m_dVel[m_currentVelWrite],  // vel(n+1/2) = vel(n) + f(n)*dt/2
			m_numParticles,
			m_dt,
			m_dt/2.0,
			1,
			m_simTime + m_dt/2.0,
			m_simparams.xsph);
	// At tis point:
	//  m_dPos[m_currentPosRead] = pos(n)
	//  m_dVel[m_currentVelRead] =  vel(n)
	//  m_dForces = f(n)
	//  m_dPos[m_currentPosWrite] = pos(n+1/2) = pos(n) + vel(n)*dt/2
	//  m_dVel[m_currentVelWrite] =  vel(n+1/2) = vel(n) + f(n)*dt/2

	if (timing) {
		cudaEventRecord(stop_euler, 0);
		cudaEventSynchronize(stop_euler);
		cudaEventElapsedTime(&m_timingInfo.timeEuler, start_euler, stop_euler);
		m_timingInfo.timeEuler *= 1e-3;
		cudaEventDestroy(start_euler);
		cudaEventDestroy(stop_euler);
		}

	// setting moving boundaries data if necessary
	if (m_simparams.mbcallback) {
		float4* hMbData = m_problem->get_mbdata(m_simTime + m_dt, m_dt/2.0, m_iter == 0);
		if (hMbData)
			setmbdata(hMbData, m_mbDataSize);
		}
	if (m_simparams.gcallback) {
		m_physparams.gravity = m_problem->g_callback(m_simTime);
		setgravity(m_physparams.gravity);
	}

	dt2 = forces(   m_dPos[m_currentPosWrite],  // pos(n+1/2)
					m_dVel[m_currentVelWrite],  // vel(n+1/2)
					m_dForces,					// f(n+1/2)
					m_dRbForces,
					m_dRbTorques,
					m_dXsph,
					m_dInfo[m_currentInfoRead],
					m_dParticleHash,
					m_dCellStart,
					m_dNeibsList,
					m_numParticles,
					m_simparams.slength,
					m_dt,
					m_simparams.dtadapt,
					m_simparams.dtadaptfactor,
					m_simparams.xsph,
					m_simparams.kerneltype,
					m_influenceRadius,
					m_simparams.visctype,
					m_physparams.visccoeff,
					m_dCfl,
					m_dTempCfl,
					m_numPartsFmax,
					m_dTau,
					m_simparams.sph_formulation,
					m_simparams.boundarytype,
					m_simparams.usedem);
	// At this point forces = f(pos(n+1/2), vel(n+1/2))

	if (m_simparams.numODEbodies) {
		reduceRbForces(m_dRbForces, m_dRbTorques, m_dRbNum, m_hRbLastIndex, m_hRbTotalForce,
						m_hRbTotalTorque, m_simparams.numODEbodies, m_numBodiesParticles);

		m_problem->ODE_bodies_timestep(m_hRbTotalForce, m_hRbTotalTorque, 2, m_dt, cg, trans, rot);
		setforcesrbcg(cg, m_simparams.numODEbodies);
		seteulerrbcg(cg, m_simparams.numODEbodies);
		seteulerrbtrans(trans, m_simparams.numODEbodies);
		seteulerrbsteprot(rot, m_simparams.numODEbodies);
	}


	euler(  m_dPos[m_currentPosRead],   // pos(n)
			m_dParticleHash,
			m_dVel[m_currentVelRead],   // vel(n)
			m_dInfo[m_currentInfoRead], //particleInfo
			m_dForces,					// f(n+1/2)
			m_dXsph,
			m_dPos[m_currentPosWrite],  // pos(n+1) = pos(n) + velc(n+1/2)*dt
			m_dVel[m_currentVelWrite],  // vel(n+1) = vel(n) + f(n+1/2)*dt
			m_numParticles,
			m_dt,
			m_dt/2.0,
			2,
			m_simTime + m_dt,
			m_simparams.xsph);
	// At this point:
	//  m_dPos[m_currentPosRead] = pos(n)
	//  m_dVel[m_currentVelRead] =  vel(n)
	//  m_dForces = f(n+1/2)
	//  m_dPos[m_currentPosWrite] = pos(n+1) = pos(n) + velc(n+1/2)*dt
	//  m_dVel[m_currentVelWrite] =  vel(n+1) = vel(n) + f(n+1/2)*dt

	std::swap(m_currentPosRead, m_currentPosWrite);
	std::swap(m_currentVelRead, m_currentVelWrite);

	// Free surface detection (Debug)
	//savenormals();

	float oldtime = m_simTime;
	m_simTime += m_dt;
	m_iter++;

	if (oldtime == m_simTime) {
		fprintf(stderr, "[%g] WARNING: timestep %g too small: time is standing still\n",
				m_simTime, m_dt);
	}
	if (m_simparams.dtadapt) {
		m_dt = min(dt1, dt2);   // next time step value
		if (!m_dt) {
			throw DtZeroException(m_simTime, m_dt);
		} else if (m_dt < FLT_EPSILON) {
			fprintf(stderr, "[%g] WARNING! new timestep %g too small!\n",
					m_simTime, m_dt);
		}
	}

	if (timing) {
		m_timingInfo.meanTimeInteract = (m_timingInfo.meanTimeInteract*(m_iter - 1) + m_timingInfo.timeInteract)/m_iter;
		m_timingInfo.meanTimeEuler = (m_timingInfo.meanTimeEuler*(m_iter - 1) + m_timingInfo.timeEuler)/m_iter;
		}

	m_timingInfo.dt = m_dt;
	m_timingInfo.t = m_simTime;
	m_timingInfo.iterations++;

	return m_timingInfo;
}


/****************************************************************************************************/
// Utility function provided for debug purpose
/****************************************************************************************************/
void
ParticleSystem::saveneibs()
{
	getArray(POSITION);
	getArray(NEIBSLIST);
	std::string fname;
	fname = m_problem->get_dirname() + "/neibs.txt";
	FILE *fp = fopen(fname.c_str(),"w");
	for (uint index = 0; index < m_numParticles; index++) {
		float3 pos;
		pos.x = m_hPos[index].x;
		pos.y = m_hPos[index].y;
		pos.z = m_hPos[index].z;

		// TODO: fix it for neib index interleave and new neibdata
		for(uint i = index*m_simparams.maxneibsnum; i < index*m_simparams.maxneibsnum + m_simparams.maxneibsnum; i++) {
			//uint neib_index = m_hNeibsList[i];
			uint neib_index = 0;

			if (neib_index == 0xffffffff) break;

			int3 periodic = make_int3(0);

			float3 neib_pos;
			neib_pos.x = m_hPos[neib_index].x;
			neib_pos.y = m_hPos[neib_index].y;
			neib_pos.z = m_hPos[neib_index].z;

			float3 relPos;
			relPos.x = pos.x - neib_pos.x;
			relPos.y = pos.y - neib_pos.y;
			relPos.z = pos.z - neib_pos.z;
			float3 relPos2;
			relPos2 = relPos + periodic*m_physparams.dispvect;

			fprintf(fp, "%d\t%f\t%f\t%f\t", index, pos.x, pos.y, pos.z);
			fprintf(fp, "%d\t%f\t%f\t%f\t", neib_index, relPos.x, relPos.y, relPos.z);
			fprintf(fp, "%d\t%d\t%d\t%f\t%f\t%f\n", periodic.x, periodic.y, periodic.z, relPos2.x, relPos2.y, relPos2.z);
			}
		}
	fclose(fp);
}


//TODO: fix for new neib data
void
ParticleSystem::savehash()
{
	getArray(POSITION);
	getArray(HASH);
	std::string fname;
	fname = m_problem->get_dirname() + "/hash.txt";
	FILE *fp = fopen(fname.c_str(),"w");
	for (uint index = 0; index < m_numParticles; index++) {
		float3 pos;
		pos.x = m_hPos[index].x;
		pos.y = m_hPos[index].y;
		pos.z = m_hPos[index].z;
		uint hash = m_hParticleHash[index];
		int3 gridPos;
		gridPos.x = floor((pos.x - m_worldOrigin.x) / m_cellSize.x);
		gridPos.y = floor((pos.y - m_worldOrigin.y) / m_cellSize.y);
		gridPos.z = floor((pos.z - m_worldOrigin.z) / m_cellSize.z);
		gridPos.x = std::max(0, std::min(gridPos.x, (int) m_gridSize.x-1));
		gridPos.y = std::max(0, std::min(gridPos.y, (int) m_gridSize.y-1));
		gridPos.z = std::max(0, std::min(gridPos.z, (int) m_gridSize.z-1));
		uint chash = gridPos.z*m_gridSize.y*m_gridSize.x + gridPos.y*m_gridSize.x + gridPos.x;

		fprintf(fp, "%d\t%f\t%f\t%f\t%d\t%d\n", index, pos.x, pos.y, pos.z,
					hash, chash);
		}
	fclose(fp);

}

void
ParticleSystem::saveindex()
{
	getArray(POSITION);
	getArray(PARTINDEX);
	std::string fname;
	fname = m_problem->get_dirname() + "/sortedindex.txt";
	FILE *fp = fopen(fname.c_str(),"w");
	for (uint index = 0; index < m_numParticles; index++) {
		float3 pos;
		uint sindex = m_hParticleIndex[index];
		pos.x = m_hPos[sindex].x;
		pos.y = m_hPos[sindex].y;
		pos.z = m_hPos[sindex].z;
		int3 gridPos;
		gridPos.x = floor((pos.x - m_worldOrigin.x) / m_cellSize.x);
		gridPos.y = floor((pos.y - m_worldOrigin.y) / m_cellSize.y);
		gridPos.z = floor((pos.z - m_worldOrigin.z) / m_cellSize.z);
		gridPos.x = std::max(0, std::min(gridPos.x, (int) m_gridSize.x-1));
		gridPos.y = std::max(0, std::min(gridPos.y, (int) m_gridSize.y-1));
		gridPos.z = std::max(0, std::min(gridPos.z, (int) m_gridSize.z-1));
		uint chash = gridPos.z*m_gridSize.y*m_gridSize.x + gridPos.y*m_gridSize.x + gridPos.x;

		fprintf(fp, "%d\t%d\t%f\t%f\t%f\t%d\n", index, sindex, pos.x, pos.y, pos.z,
					 chash);
		}
	fclose(fp);

}

void
ParticleSystem::savesorted()
{
	getArray(POSITION);
	getArray(PARTINDEX);
	std::string fname;
	fname = m_problem->get_dirname() + "/sortedparts.txt";
	FILE *fp = fopen(fname.c_str(),"w");
	for (uint index = 0; index < m_numParticles; index++) {
		float3 pos;
		uint sindex = m_hParticleIndex[index];
		pos.x = m_hPos[index].x;
		pos.y = m_hPos[index].y;
		pos.z = m_hPos[index].z;
		int3 gridPos;
		gridPos.x = floor((pos.x - m_worldOrigin.x) / m_cellSize.x);
		gridPos.y = floor((pos.y - m_worldOrigin.y) / m_cellSize.y);
		gridPos.z = floor((pos.z - m_worldOrigin.z) / m_cellSize.z);
		gridPos.x = std::max(0, std::min(gridPos.x, (int) m_gridSize.x-1));
		gridPos.y = std::max(0, std::min(gridPos.y, (int) m_gridSize.y-1));
		gridPos.z = std::max(0, std::min(gridPos.z, (int) m_gridSize.z-1));
		uint chash = gridPos.z*m_gridSize.y*m_gridSize.x + gridPos.y*m_gridSize.x + gridPos.x;

		fprintf(fp, "%d\t%d\t%f\t%f\t%f\t%d\n", index, sindex, pos.x, pos.y, pos.z,
					 chash);
		}
	fclose(fp);

}

void
ParticleSystem::savecellstartend()
{
	getArray(POSITION);
	getArray(CELLSTART);
	getArray(CELLEND);
	std::string fname;
	fname = m_problem->get_dirname() + "/cellstartend.txt";
	FILE *fp = fopen(fname.c_str(),"w");
	for (uint index = 0; index < m_nGridCells; index++) {
		uint cstart = m_hCellStart[index];
		uint cend = m_hCellEnd[index];

		fprintf(fp, "%d\t%d\t%d\n", index, cstart, cend);
		}
	fclose(fp);
}

// Free surface detection
void
ParticleSystem::savenormals()
{
	getArray(INFO);
	std::string fname;
	fname = m_problem->get_dirname() + "/normals.txt";
	FILE *fp = fopen(fname.c_str(),"w");
	for (uint index = 0; index < m_numParticles; index++) {
		float4 normal = m_hNormals[index];

		fprintf(fp, "%d\t%f\t%f\t%f\t%f\n", index, normal.x, normal.y, normal.z, normal.w);
		}
	fclose(fp);


}

void
ParticleSystem::reducerbforces(void)
{
	CUDA_SAFE_CALL(cudaMemcpy((void *) m_hRbForces, (void *) m_dRbForces,
		m_numBodiesParticles*sizeof(float4), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy((void *) m_hRbTorques, (void *) m_dRbTorques,
		m_numBodiesParticles*sizeof(float4), cudaMemcpyDeviceToHost));

	int firstindex = 0;
	int lastindex = 0;
	for (int i = 0; i < m_simparams.numODEbodies; i++) {
		lastindex = m_hRbLastIndex[i];
		float4 force = make_float4(0.0f);
		float4 torque = make_float4(0.0f);
		for (int j = firstindex; j <= lastindex; j++) {
			force += m_hRbForces[j];
			torque += m_hRbTorques[j];
		}
		m_hRbTotalForce[i] = as_float3(force);
		m_hRbTotalTorque[i] = as_float3(torque);

		firstindex = lastindex + 1;
	}
}
/****************************************************************************************************/
