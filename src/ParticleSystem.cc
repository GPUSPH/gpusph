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
#include "Writer.h"
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
	"Normals",
	"Boundary Elements",
	"Gradient Gamma",
	"Vertices",
	"Pressure",
	"Turbulent Kinetic Energy [k]",
	"Turbulent Dissipation Rate [e]",
	"Eddy Viscosity",
	"(invalid)"
};

ParticleSystem::ParticleSystem(Problem *problem) :
	m_problem(problem),
	m_neiblist_built(false),
	m_physparams(problem->get_physparams()),
	m_simparams(problem->get_simparams()),
	m_simTime(0.0),
	m_iter(0),
	m_currentPosRead(0),
	m_currentPosWrite(1),
	m_currentVelRead(0),
	m_currentVelWrite(1),
	m_currentInfoRead(0),
	m_currentInfoWrite(1),
	m_currentBoundElementRead(0),
	m_currentBoundElementWrite(1),
	m_currentGradGammaRead(0),
	m_currentGradGammaWrite(1),
	m_currentVerticesRead(0),
	m_currentVerticesWrite(1),
	m_currentPressureRead(0),
	m_currentPressureWrite(1),
	m_currentTKERead(0),
	m_currentTKEWrite(1),
	m_currentEpsRead(0),
	m_currentEpsWrite(1),
	m_currentTurbViscRead(0),
	m_currentTurbViscWrite(1)
{
	m_worldOrigin = problem->get_worldorigin();
	m_worldSize = problem->get_worldsize();
	m_writerType = problem->get_writertype();

	m_influenceRadius = m_simparams->kernelradius*m_simparams->slength;
	m_nlInfluenceRadius = m_influenceRadius*m_simparams->nlexpansionfactor;
	m_nlSqInfluenceRadius = m_nlInfluenceRadius*m_nlInfluenceRadius;

	m_gridSize.x = (uint) (m_worldSize.x / m_influenceRadius);
	m_gridSize.y = (uint) (m_worldSize.y / m_influenceRadius);
	m_gridSize.z = (uint) (m_worldSize.z / m_influenceRadius);

	m_nGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
	m_nSortingBits = ceil(log2((float) m_nGridCells)/4.0)*4;

	m_cellSize.x = m_worldSize.x / m_gridSize.x;
	m_cellSize.y = m_worldSize.y / m_gridSize.y;
	m_cellSize.z = m_worldSize.z / m_gridSize.z;

	m_dt = m_simparams->dt;

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
	m_device = checkCUDA(problem->get_options());
	printf("\nCUDA initialized\n");

#define MEGABYTE (1024.0*1024.0)
#define GIGABYTE (MEGABYTE*1024)

	printf("\t%u multiprocessors, %zu (%g%s) global memory\n",
			m_device.multiProcessorCount,
			m_device.totalGlobalMem,
			m_device.totalGlobalMem > GIGABYTE ?
			m_device.totalGlobalMem/GIGABYTE :
			m_device.totalGlobalMem/MEGABYTE,
			m_device.totalGlobalMem > GIGABYTE ?
			"GB" : "MB");
	printf("\t%u threads, %zu (%g%s) shared memory per MP\n",
			m_device.maxThreadsPerMultiProcessor,
			m_device.sharedMemPerBlock,
			m_device.sharedMemPerBlock/1024.0, "kB");

	setPhysParams();

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

	writeSummary();
}


void
ParticleSystem::allocate(uint numParticles)
{
	if (numParticles >= MAXPARTICLES) {
		fprintf(stderr, "Cannot handle %u > %u particles, sorry\n",
				numParticles, MAXPARTICLES);
		exit(1);
	}

	if (m_simparams->maxneibsnum % NEIBINDEX_INTERLEAVE != 0) {
		fprintf(stderr, "The maximum number of neibs per particle (%u) should be a multiple of NEIBINDEX_INTERLEAVE (%u)\n",
				m_simparams->maxneibsnum, NEIBINDEX_INTERLEAVE);
		exit(1);
	}
	m_numParticles = numParticles;
	m_timingInfo.numParticles = numParticles;

	// allocate host storage
	const uint memSize = sizeof(float)*m_numParticles;
	const uint memSize2 = sizeof(float2)*m_numParticles;
	const uint memSize3 = sizeof(float3)*m_numParticles;
	const uint memSize4 = sizeof(float4)*m_numParticles;
	const uint infoSize = sizeof(particleinfo)*m_numParticles;
	const uint vinfoSize = sizeof(vertexinfo)*m_numParticles;
	const uint hashSize = sizeof(hashKey)*m_numParticles;
	const uint idxSize = sizeof(uint)*m_numParticles;
	const uint gridcellSize = sizeof(uint)*m_nGridCells;
	const uint neibslistSize = sizeof(uint)*m_simparams->maxneibsnum*(m_numParticles/NEIBINDEX_INTERLEAVE + 1)*NEIBINDEX_INTERLEAVE;

	uint memory = 0;

	m_hPos = new float4[m_numParticles];
	memset(m_hPos, 0, memSize4);
	memory += memSize4;

	m_hVel = new float4[m_numParticles];
	memset(m_hVel, 0, memSize4);
	memory += memSize4;

	m_hInfo = new particleinfo[m_numParticles];
	memset(m_hInfo, 0, infoSize);
	memory += infoSize;

	m_hEnergy = new float4[m_physparams->numFluids];
	memset(m_hInfo, 0, sizeof(float4)*m_physparams->numFluids); //TODO Should it be m_hEnergy instead of m_hInfo?
	memory += sizeof(float4)*m_physparams->numFluids;

	m_hVort = NULL;
	if (m_simparams->vorticity) {
		m_hVort = new float3[m_numParticles];
		memory += memSize3;
		}

	m_hGradGamma = new float4[m_numParticles];
	memset(m_hGradGamma, 0, memSize4);
	memory += memSize4;
	
	m_hBoundElement = new float4[m_numParticles];
	memset(m_hBoundElement, 0, memSize4);
	memory += memSize4;
	
	m_hVertices = new vertexinfo[m_numParticles];
	memset(m_hVertices, 0, vinfoSize);
	memory += vinfoSize;

	m_hPressure = new float[m_numParticles];
	memset(m_hPressure, 0, memSize);
	memory += memSize;

	m_hTKE = new float[m_numParticles];
	memset(m_hTKE, 0, memSize);
	memory += memSize;

	m_hEps = new float[m_numParticles];
	memset(m_hEps, 0, memSize);
	memory += memSize;

	m_hTurbVisc = new float[m_numParticles];
	memset(m_hTurbVisc, 0, memSize);
	memory += memSize;

	m_hForces = new float4[m_numParticles];
	memset(m_hForces, 0, memSize4);
	memory += memSize4;


#ifdef _DEBUG_
	m_hParticleHash = new hashKey[m_numParticles];
	memset(m_hParticleHash, 0, hashSize);
	memory += hashSize;

	m_hParticleIndex = new uint[m_numParticles];
	memset(m_hParticleIndex, 0, idxSize);
	memory += idxSize;

	m_hCellStart = new uint[m_nGridCells];
	memset(m_hCellStart, 0, gridcellSize);
	memory += gridcellSize;

	m_hCellEnd = new uint[m_nGridCells];
	memset(m_hCellEnd, 0, gridcellSize);
	memory += gridcellSize;

	m_hNeibsList = new uint[m_simparams->maxneibsnum*(m_numParticles/NEIBINDEX_INTERLEAVE + 1)*NEIBINDEX_INTERLEAVE];
	memset(m_hNeibsList, 0xffff, neibslistSize);
	memory += neibslistSize;
#endif

	// Free surface detection
	m_hNormals = NULL;
	if (m_simparams->savenormals) {
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

	// Free surface detection
	if (m_simparams->savenormals) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dNormals, memSize4));
		memory += memSize4;
	}

	if (m_simparams->vorticity) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dVort, memSize3));
		memory += memSize3;
	}

	if (m_simparams->visctype == SPSVISC) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTau[0], memSize2));
		memory += memSize2;

		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTau[1], memSize2));
		memory += memSize2;

		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTau[2], memSize2));
		memory += memSize2;
	}
	
	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dGradGamma[0], memSize4));
	memory += memSize4;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dGradGamma[1], memSize4));
	memory += memSize4;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dBoundElement[0], memSize4));
	memory += memSize4;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dBoundElement[1], memSize4));
	memory += memSize4;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dVertices[0], vinfoSize));
	memory += vinfoSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dVertices[1], vinfoSize));
	memory += vinfoSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dPressure[0], memSize));
	memory += memSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dPressure[1], memSize));
	memory += memSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTKE[0], memSize));
	memory += memSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTKE[1], memSize));
	memory += memSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dEps[0], memSize));
	memory += memSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dEps[1], memSize));
	memory += memSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTurbVisc[0], memSize));
	memory += memSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTurbVisc[1], memSize));
	memory += memSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dParticleHash, hashSize));
	memory += hashSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dParticleIndex, idxSize));
	memory += idxSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dInversedParticleIndex, hashSize));
	memory += hashSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dCellStart, gridcellSize));
	memory += gridcellSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dCellEnd, gridcellSize));
	memory += gridcellSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dNeibsList, neibslistSize));
	memory += neibslistSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dNewNumParticles, sizeof(uint)));
	memory += sizeof(uint);

	// Allocate storage for rigid bodies forces and torque computation
	if (m_simparams->numbodies) {
		m_numBodiesParticles = m_problem->get_bodies_numparts();
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
		m_hRbLastIndex = new uint[m_simparams->numbodies];
		m_hRbTotalForce = new float3[m_simparams->numbodies];
		m_hRbTotalTorque = new float3[m_simparams->numbodies];

		rbfirstindex[0] = 0;
		for (int i = 1; i < m_simparams->numbodies; i++) {
			rbfirstindex[i] = rbfirstindex[i - 1] + m_problem->get_body_numparts(i - 1);
		}
		setforcesrbstart(rbfirstindex, m_simparams->numbodies);

		int offset = 0;
		for (int i = 0; i < m_simparams->numbodies; i++) {
			m_hRbLastIndex[i] = m_problem->get_body_numparts(i) - 1 + offset;
			for (int j = 0; j < m_problem->get_body_numparts(i); j++) {
				rbnum[offset + j] = i;
			}
			offset += m_problem->get_body_numparts(i);
		}
		size_t  size = m_numBodiesParticles*sizeof(uint);
		CUDA_SAFE_CALL(cudaMemcpy((void *) m_dRbNum, (void*) rbnum, size, cudaMemcpyHostToDevice));

		delete[] rbnum;
	}

	if (m_simparams->dtadapt) {
		m_numPartsFmax = getNumPartsFmax(numParticles);
		const uint fmaxTableSize = m_numPartsFmax*sizeof(float);

		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dCfl, fmaxTableSize));
		CUDA_SAFE_CALL(cudaMemset(m_dCfl, 0, fmaxTableSize));

		if(m_simparams->boundarytype == MF_BOUNDARY) {
			CUDA_SAFE_CALL(cudaMalloc((void**)&m_dCflGamma, fmaxTableSize));
			CUDA_SAFE_CALL(cudaMemset(m_dCflGamma, 0, fmaxTableSize));
		}

		const uint tempCflSize = getFmaxTempStorageSize(m_numPartsFmax);
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTempCfl, tempCflSize));
		CUDA_SAFE_CALL(cudaMemset(m_dTempCfl, 0, tempCflSize));

		memory += fmaxTableSize;
		}

	// number of blocks to be used in parallel reductions
	size_t blocks = 6*m_device.multiProcessorCount;

	/*	threads per blocks in a parallel reduction: try to get
		the largest possible power-of-two blocksize such that
		two blocks can fit in a single MP */
	size_t blocksize = 32;
	while (blocksize*2 < m_device.maxThreadsPerMultiProcessor)
		blocksize*=2;

	/*	maximum amount of global memory needed in reductions:
		this value should be kept up to date to fit the largest
		possible reduction used during the simulation */
	// presently: two float4 value per fluid type (allows Kahan summation)
	size_t bufsize = 2*sizeof(float4)*blocks*m_physparams->numFluids;

	void *buff = NULL;
	CUDA_SAFE_CALL(cudaMalloc(&buff, bufsize));
	memory += bufsize;

	printf("%zuB reserved for reductions\n", bufsize);
	printf("\treductions will use %zu blocks, %zu threads max each\n",
		blocks, blocksize);
	set_reduction_params(buff, blocks, blocksize, m_device.sharedMemPerBlock);

	// Allocating, reading and copying DEM
	if (m_simparams->usedem) {
		printf("Using DEM\n");
		int2 dem_rc;
		dem_rc.x = m_problem->get_dem_ncols();
		dem_rc.y = m_problem->get_dem_nrows();
		printf("cols = %d\trows =% d\n", dem_rc.x, dem_rc.y);
		setDemTexture(m_problem->get_dem(), dem_rc.x, dem_rc.y);
		}
	printf("Number of fluids: %d\n",m_physparams->numFluids);
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
	switch (m_simparams->visctype) {
		case ARTVISC:
			m_physparams->visccoeff = m_physparams->artvisccoeff;
			break;

		case KINEMATICVISC:
		case SPSVISC:
			m_physparams->visccoeff = 4.0*m_physparams->kinematicvisc;
			break;

		case DYNAMICVISC:
			m_physparams->visccoeff = m_physparams->kinematicvisc;
			break;
	}

	// Setting kernels and kernels derivative factors

	setforcesconstants(m_simparams, m_physparams);
	seteulerconstants(m_physparams);
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

	while(i < m_physparams->numFluids) {
		fprintf(summary,"fluid #%u \n", i);
		fprintf(summary, "\trho0 = %g\n", m_physparams->rho0[i]);
		fprintf(summary, "\tb = %g\n", m_physparams->bcoeff[i]);
		fprintf(summary, "\tgamma = %g\n", m_physparams->gammacoeff[i]);
		fprintf(summary, "\tsscoeff = %g\n", m_physparams->sscoeff[i]);
		fprintf(summary, "\tsspowercoeff = %g\n", m_physparams->sspowercoeff[i]);
		fprintf(summary, "\tsound speed = %g\n", m_problem->soundspeed(m_physparams->rho0[i],i));
		++i;
		}

#define g m_physparams->gravity
	fprintf(summary, "gravity = (%g, %g, %g) [%g]\n", g.x, g.y, g.z, length(g));
#undef g

	fprintf(summary, "%s boundary parameters:\n", BoundaryName[m_simparams->boundarytype]);
	switch (m_simparams->boundarytype) {
		case LJ_BOUNDARY:
			fprintf(summary, "\td = %g\n", m_physparams->dcoeff);
			fprintf(summary, "\tp1 = %g\n", m_physparams->p1coeff);
			fprintf(summary, "\tp2 = %g\n", m_physparams->p2coeff);
			break;
		case MK_BOUNDARY:
			fprintf(summary, "\tK = %g\n", m_physparams->MK_K);
			fprintf(summary, "\td = %g\n", m_physparams->MK_d);
			fprintf(summary, "\tbeta = %g\n", m_physparams->MK_beta);
			break;
	}

	fprintf(summary, "r0 = %g\n", m_physparams->r0);
	float visccoeff = m_physparams->visccoeff;
	fprintf(summary,"Viscosity\n");
	switch (m_simparams->visctype) {
		case ARTVISC:
			fprintf(summary, "\tArtificial viscosity: artvisccoeff = %g (viscoeff=%g = artvisccoeff)\n",
					m_physparams->artvisccoeff, visccoeff);
			break;

		case KINEMATICVISC:
			fprintf(summary, "\tKinematic viscosity: kinematicvisc = %g m^2/s (viscoeff=%g = 4*kinematicvisc)\n",
					m_physparams->kinematicvisc, visccoeff);
			break;

		case DYNAMICVISC:
			fprintf(summary, "\tDynamic viscosity: kinematicvisc = %g m^2/s (viscoeff=%g = kinematicvisc)\n",
					m_physparams->kinematicvisc, visccoeff);
			break;

		case SPSVISC:
			fprintf(summary, "\tSPS + kinematic viscosity: kinematicvisc = %g m^2/s (viscoeff=%g = kinematicvisc)\n",
					m_physparams->kinematicvisc, visccoeff);
			fprintf(summary, "\tSmagFactor = %g\n", m_physparams->smagfactor);
			fprintf(summary, "\tkSPSFactor = %g\n", m_physparams->kspsfactor);
			break;
		case KEPS:
			fprintf(summary, "\tk-e model: kinematicvisc = %g m^2/s\n",
					m_physparams->kinematicvisc);
			break;
		}
	fprintf(summary, "espartvisc = %g\n", m_physparams->epsartvisc);
	fprintf(summary, "epsxsph = %g\n", m_physparams->epsxsph);
	if (m_simparams->periodicbound) {
		fprintf(summary, "Periodic boundary parameters (disp vect, min and max limit) used when x,y or z periodic boundary is set\n");
		fprintf(summary, "disp vect = (%g, %g, %g)\n", m_physparams->dispvect.x, m_physparams->dispvect.y, m_physparams->dispvect.z);
		fprintf(summary, "min limit = (%g, %g, %g)\n", m_physparams->minlimit.x, m_physparams->minlimit.y, m_physparams->minlimit.z);
		fprintf(summary, "max limit = (%g, %g, %g)\n", m_physparams->maxlimit.x, m_physparams->maxlimit.y, m_physparams->maxlimit.z);
		}
	if (m_simparams->usedem) {
		fprintf(summary, "DEM resolution ew = %g, ns = %g\n", m_physparams->ewres, m_physparams->nsres);
		fprintf(summary, "Displacement for normal computing dx = %g, dy = %g\n", m_physparams->demdx, m_physparams->demdy);
		fprintf(summary, "DEM zmin = %g\n", m_physparams->demzmin);
		}

	fflush(summary);
}


void
ParticleSystem::printSimParams(FILE *summary)
{
	if (!summary)
		summary = stdout;

	fprintf(summary, "\nSimulation parameters:\n");
	fprintf(summary, "slength = %g\n", m_simparams->slength);
	fprintf(summary, "kernelradius = %g\n", m_simparams->kernelradius);
	fprintf(summary, "initial dt = %g\n", m_simparams->dt);
	fprintf(summary, "simulation end time = %g\n", m_simparams->tend);
	fprintf(summary, "neib list construction every %d iteration\n", m_simparams->buildneibsfreq);
	fprintf(summary, "Shepard filter every %d iteration\n", m_simparams->shepardfreq);
	fprintf(summary, "MLS filter every %d iteration\n", m_simparams->mlsfreq);
	fprintf(summary, "Ferrari correction = %g\n", m_simparams->ferrari);
	fprintf(summary, "adaptive time step = %d\n", m_simparams->dtadapt);
	fprintf(summary, "safety factor for adaptive time step = %f\n", m_simparams->dtadaptfactor);
	fprintf(summary, "xsph correction = %d\n", m_simparams->xsph);
	fprintf(summary, "SPH formulation = %d\n", m_simparams->sph_formulation);
	fprintf(summary, "viscosity type = %d (%s)\n", m_simparams->visctype, ViscosityName[m_simparams->visctype]);
	fprintf(summary, "moving boundary velocity callback function = %d (0 none)\n", m_simparams->mbcallback);
	if (m_simparams->mbcallback)
		fprintf(summary, "\tnumber of moving boundaries = %d\n", m_problem->m_mbnumber);
	fprintf(summary, "variable gravity callback function = %d\n",m_simparams->gcallback);
	fprintf(summary, "periodic boundary = %s\n", m_simparams->periodicbound ? "true" : "false");
	fprintf(summary, "using DEM = %d\n", m_simparams->usedem);
	fprintf(summary, "number of rigid bodies = %d\n", m_simparams->numbodies);

	fflush(summary);
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
	delete [] m_hVel;
	delete [] m_hInfo;
	if (m_simparams->vorticity) {
		delete [] m_hVort;
		}
	if (m_hVertices)
		delete [] m_hVertices;
	if (m_hGradGamma)
		delete [] m_hGradGamma;
	if (m_hBoundElement)
		delete [] m_hBoundElement;
	if (m_hPressure)
		delete [] m_hPressure;

	delete [] m_hForces;
#ifdef _DEBUG_
	delete [] m_hNeibsList;
	delete [] m_hParticleHash;
	delete [] m_hParticleIndex;
	delete [] m_hCellStart;
	delete [] m_hCellEnd;
#endif
	if (m_simparams->savenormals)
		delete [] m_hNormals;

	delete m_writer;

	unset_reduction_params();

	CUDA_SAFE_CALL(cudaFree(m_dForces));
	CUDA_SAFE_CALL(cudaFree(m_dXsph));

	if (m_simparams->visctype == SPSVISC) {
		CUDA_SAFE_CALL(cudaFree(m_dTau[0]));
		CUDA_SAFE_CALL(cudaFree(m_dTau[1]));
		CUDA_SAFE_CALL(cudaFree(m_dTau[2]));
		}

	if (m_simparams->vorticity) {
		CUDA_SAFE_CALL(cudaFree(m_dVort));
		}

	CUDA_SAFE_CALL(cudaFree(m_dPos[0]));
	CUDA_SAFE_CALL(cudaFree(m_dPos[1]));
	CUDA_SAFE_CALL(cudaFree(m_dVel[0]));
	CUDA_SAFE_CALL(cudaFree(m_dVel[1]));
	CUDA_SAFE_CALL(cudaFree(m_dInfo[0]));
	CUDA_SAFE_CALL(cudaFree(m_dInfo[1]));
	
	CUDA_SAFE_CALL(cudaFree(m_dGradGamma[0]));
	CUDA_SAFE_CALL(cudaFree(m_dGradGamma[1]));
	CUDA_SAFE_CALL(cudaFree(m_dBoundElement[0]));
	CUDA_SAFE_CALL(cudaFree(m_dBoundElement[1]));
	CUDA_SAFE_CALL(cudaFree(m_dVertices[0]));
	CUDA_SAFE_CALL(cudaFree(m_dVertices[1]));
	CUDA_SAFE_CALL(cudaFree(m_dPressure[0]));
	CUDA_SAFE_CALL(cudaFree(m_dPressure[1]));
	
	// Free surface detection
	if (m_simparams->savenormals)
		CUDA_SAFE_CALL(cudaFree(m_dNormals));

	if (m_simparams->numbodies) {
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
	CUDA_SAFE_CALL(cudaFree(m_dInversedParticleIndex));
	CUDA_SAFE_CALL(cudaFree(m_dCellStart));
	CUDA_SAFE_CALL(cudaFree(m_dCellEnd));
	CUDA_SAFE_CALL(cudaFree(m_dNeibsList));

	if (m_simparams->usedem)
		releaseDemTexture();

	if (m_simparams->dtadapt) {
		CUDA_SAFE_CALL(cudaFree(m_dCfl));
		CUDA_SAFE_CALL(cudaFree(m_dTempCfl));

		if (m_simparams->boundarytype == MF_BOUNDARY)
			CUDA_SAFE_CALL(cudaFree(m_dCflGamma));
		}

	printf("GPU and CPU memory released\n\n");
}


// TODO: DEBUG, testpoints, freesurface and size_t
void*
ParticleSystem::getArray(ParticleArray array, bool need_write)
{
	void*   hdata = 0;
	void*   ddata = 0;
	long	size;

	/* when writing testpoints and/or surface particles on an initial save,
	 * we have to ensure that the neiblist has been built; this must be done
	 * before ANY array is retrieved, otherwise the information might be out
	 * of sync for arrays taken before/after the sort done by the neiblist
	 * construction */

	if (need_write
			&& (m_simparams->testpoints || m_simparams->surfaceparticle)
			&& !m_neiblist_built)
		buildNeibList(false);

	switch (array) {
		default:
		case POSITION:
			size = m_numParticles*sizeof(float4);
			hdata = (void*) m_hPos;
			ddata = (void*) m_dPos[m_currentPosRead];
			break;

		case VELOCITY:
			{
			//Testpoints
			if (need_write && m_simparams->testpoints) {
				testpoints(	m_dPos[m_currentPosRead],
							m_dVel[m_currentVelRead],
							m_dInfo[m_currentInfoRead],
							m_dNeibsList,
							m_numParticles,
							m_simparams->slength,
							m_simparams->kerneltype,
							m_influenceRadius,
							m_simparams->periodicbound);
				} // if need_write && m_simparams->testpoints

			size = m_numParticles*sizeof(float4);
			hdata = (void*) m_hVel;
			ddata = (void*) m_dVel[m_currentVelRead];
			}
			break;

		case INFO:
			// Free surface detection
			if (need_write && m_simparams->surfaceparticle) {
				surfaceparticle( m_dPos[m_currentPosRead],
								 m_dVel[m_currentVelRead],
						         m_dNormals,
								 m_dInfo[m_currentInfoRead],
								 m_dInfo[m_currentInfoWrite],
								 m_dNeibsList,
								 m_numParticles,
								 m_simparams->slength,
								 m_simparams->kerneltype,
								 m_influenceRadius,
								 m_simparams->periodicbound,
								 m_simparams->savenormals);
				std::swap(m_currentInfoRead, m_currentInfoWrite);
				} // if need_write && m_simparams->surfaceparticle

			size = m_numParticles*sizeof(particleinfo);
			hdata = (void*) m_hInfo;
			ddata = (void*) m_dInfo[m_currentInfoRead];
			break;

		case VORTICITY:
			size = m_numParticles*sizeof(float3);
			hdata = (void*) m_hVort;
			ddata = (void*) m_dVort;

			// Calling vorticity computation kernel
			vorticity(	m_dPos[m_currentPosRead],
						m_dVel[m_currentVelRead],
						m_dVort,
						m_dInfo[m_currentInfoRead],
						m_dNeibsList,
						m_numParticles,
						m_simparams->slength,
						m_simparams->kerneltype,
						m_influenceRadius,
						m_simparams->periodicbound);
			break;

		case FORCE:
			size = m_numParticles*sizeof(float4);
			hdata = (void*) m_hForces;
			ddata = (void*) m_dForces;
			break;
#ifdef _DEBUG_
		case NEIBSLIST:
			size = m_numParticles*m_simparams->maxneibsnum*sizeof(uint);
			hdata = (void*) m_hNeibsList;
			ddata = (void*) m_dNeibsList;
			break;

		case HASH:
			size = m_numParticles*sizeof(hashKey);
			hdata = (void*) m_hParticleHash;
			ddata = (void*) m_dParticleHash;
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
			// Free surface detection (Debug)
		case NORMALS:
			size = m_numParticles*sizeof(float4);
			hdata = (void*) m_hNormals;
			ddata = (void*) m_dNormals;
			break;

		case BOUNDELEMENT:
			size = m_numParticles*sizeof(float4);
			hdata = (void*) m_hBoundElement;
			ddata = (void*) m_dBoundElement[m_currentBoundElementRead];
			break;

		case GRADGAMMA:
			size = m_numParticles*sizeof(float4);
			hdata = (void*) m_hGradGamma;
			ddata = (void*) m_dGradGamma[m_currentGradGammaRead];
			break;

		case VERTICES:
			size = m_numParticles*sizeof(vertexinfo);
			hdata = (void*) m_hVertices;
			ddata = (void*) m_dVertices[m_currentVerticesRead];
			break;

		case PRESSURE:
			size = m_numParticles*sizeof(float);
			hdata = (void*) m_hPressure;
			ddata = (void*) m_dPressure[m_currentPressureRead];

		case TKE:
			size = m_numParticles*sizeof(float);
			hdata = (void*) m_hTKE;
			ddata = (void*) m_dTKE[m_currentTKERead];

		case EPSILON:
			size = m_numParticles*sizeof(float);
			hdata = (void*) m_hEps;
			ddata = (void*) m_dEps[m_currentEpsRead];

		case TURBVISC:
			size = m_numParticles*sizeof(float);
			hdata = (void*) m_hTurbVisc;
			ddata = (void*) m_dTurbVisc[m_currentTurbViscRead];
	}

	CUDA_SAFE_CALL(cudaMemcpy(hdata, ddata, size, cudaMemcpyDeviceToHost));
	return hdata;
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
			hdata = m_hPos;
			ddata = m_dPos[m_currentPosRead];
			size = m_numParticles*sizeof(float4);
			break;

		case VELOCITY:
			hdata = m_hVel;
			ddata = m_dVel[m_currentVelRead];
			size = m_numParticles*sizeof(float4);
			break;

		case INFO:
			hdata = m_hInfo;
			ddata = m_dInfo[m_currentInfoRead];
			size = m_numParticles*sizeof(particleinfo);
			break;

		case BOUNDELEMENT:
			hdata = m_hBoundElement;
			ddata = m_dBoundElement[m_currentBoundElementRead];
			size = m_numParticles*sizeof(float4);
			break;

		case GRADGAMMA:
			hdata = m_hGradGamma;
			ddata = m_dGradGamma[m_currentGradGammaRead];
			size = m_numParticles*sizeof(float4);
			break;

		case VERTICES:
			hdata = m_hVertices;
			ddata = m_dVertices[m_currentVerticesRead];
			size = m_numParticles*sizeof(vertexinfo);
			break;

		case PRESSURE:
			hdata = m_hPressure;
			ddata = m_dPressure[m_currentPressureRead];
			size = m_numParticles*sizeof(float);
			break;

		case TKE:
			hdata = m_hTKE;
			ddata = m_dTKE[m_currentTKERead];
			size = m_numParticles*sizeof(float);
			break;

		case EPSILON:
			hdata = m_hEps;
			ddata = m_dEps[m_currentEpsRead];
			size = m_numParticles*sizeof(float);
			break;

		case TURBVISC:
			hdata = m_hTurbVisc;
			ddata = m_dTurbVisc[m_currentTurbViscRead];
			size = m_numParticles*sizeof(float);
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
	//Testpoints
	m_writer->write(m_numParticles, m_hPos, m_hVel, m_hInfo, m_hVort, m_simTime, m_simparams->testpoints, m_hNormals, m_hGradGamma, m_hTKE, m_hTurbVisc);
	m_problem->mark_written(m_simTime);
	calc_energy(m_hEnergy,
		m_dPos[m_currentPosRead],
		m_dVel[m_currentVelRead],
		m_dInfo[m_currentInfoRead],
		m_numParticles,
		m_physparams->numFluids);
	m_writer->write_energy(m_simTime, m_hEnergy);
}


void
ParticleSystem::drawParts(bool show_boundary, bool show_floating, bool show_vertex, int view_mode)
{
	float minrho = m_problem->get_minrho();
	float maxrho = m_problem->get_maxrho();
	float minvel = m_problem->get_minvel();
	float maxvel = m_problem->get_maxvel();

	float minp = m_problem->pressure(minrho,0); //FIX FOR MULT-FLUID
	float maxp = m_problem->pressure(maxrho,0);

	float4* pos = m_hPos;
	float4* vel = m_hVel;
	float3* vort = m_hVort;
	particleinfo* info = m_hInfo;
	vertexinfo* vinfo = m_hVertices;

	bool show_fluid = view_mode == VM_NOFLUID? false : true;

	glPointSize(2.0);
	glBegin(GL_POINTS);
	{
		for (uint i = 0; i < m_numParticles; i++) {
			if (NOT_FLUID(info[i]) && !OBJECT(info[i]) && !VERTEX(info[i]) && show_boundary) {
				glColor3f(0.0, 1.0, 0.0);
				glVertex3fv((float*)&pos[i]);
			}
			if (OBJECT(info[i]) && show_floating) {
				glColor3f(1.0, 0.0, 0.0);
				glVertex3fv((float*)&pos[i]);
			}
			if (VERTEX(info[i]) && show_vertex) {
				glColor3f(0.3, 0.7, 0.9);
				glVertex3fv((float*)&pos[i]);
			}
			if (PROBE(info[i])) {
				glColor3f(0.0, 0.0, 0.0);
				glVertex3fv((float*)&pos[i]);
			}
			if (FLUID(info[i]) && show_fluid) {
				float v; unsigned int t;
				float ssvel = m_problem->soundspeed(vel[i].w, PART_FLUID_NUM(info[i]));
				switch (view_mode) {
					case VM_NORMAL:
					    glColor3f(0.0,0.0,1.0);
					    if (m_physparams->numFluids > 1) {
					       v = (float) PART_FLUID_NUM(info[i]);
						v /= (m_physparams->numFluids - 1);
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
					    v = length(vort[i]);
					    glColor3f(1.-(v-minvel)/(maxvel-minvel),1.0,1.0);
					    break;
				}
				glVertex3fv((float*)&pos[i]);
			}
		}

	}
	glEnd();

	//draw connections between vertex particles (i.e. triangular boundary elements)
	glBegin(GL_TRIANGLES);
	glColor3f(0.2,0.6,0.8);
	for(uint i = 0; i < m_numParticles; i++) {
		if(BOUNDARY(info[i]) && show_vertex) {
			uint i_vert1 = vinfo[i].x;
			uint i_vert2 = vinfo[i].y;
			uint i_vert3 = vinfo[i].z;
			glVertex3f(pos[i_vert1].x, pos[i_vert1].y, pos[i_vert1].z);
			glVertex3f(pos[i_vert2].x, pos[i_vert2].y, pos[i_vert2].z);
			glVertex3f(pos[i_vert3].x, pos[i_vert3].y, pos[i_vert3].z);
		}
	}
	glEnd();

	if (m_simparams->gage.size() > 0) {
		float lw;
		glGetFloatv(GL_LINE_WIDTH, &lw);
		glLineWidth(2.0);
		glBegin(GL_LINES);
		glColor3f(0,0,0);
		GageList::iterator g = m_simparams->gage.begin();
		GageList::iterator end = m_simparams->gage.end();
		while (g != end) {
			glVertex3f(g->x, g->y, m_worldOrigin.z);
			glVertex3f(g->x, g->y, m_worldSize.z);
			++g;
		}
		glEnd();
		glLineWidth(lw);
	}
}

void
ParticleSystem::buildNeibList(bool timing)
{
	cudaEvent_t start_neibslist, stop_neibslist;

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
#if HASH_KEY_SIZE >= 64
			m_dInfo[m_currentPosRead],
#endif
			m_dParticleHash,
			m_dParticleIndex,
			gridSize,
			cellSize,
			worldOrigin,
			m_numParticles);

	// hash based particle sort
	sort(m_dParticleHash, m_dParticleIndex, m_numParticles);

	//inverse array with particle indexes;
	//array with inversed indexes will be used later to reassign vertices for boundary elements
	inverseParticleIndex(m_dParticleIndex, m_dInversedParticleIndex, m_numParticles);

	reorderDataAndFindCellStart(
			m_dCellStart,					// output: cell start index
			m_dCellEnd,					// output: cell end index
			m_dPos[m_currentPosWrite],			// output: sorted positions
			m_dVel[m_currentVelWrite],			// output: sorted velocities
			m_dInfo[m_currentInfoWrite],			// output: sorted info
			m_dBoundElement[m_currentBoundElementWrite],	// output: sorted boundary elements
			m_dGradGamma[m_currentGradGammaWrite],		// output: sorted gradient gamma
			m_dVertices[m_currentVerticesWrite],		// output: sorted vertices
			m_dPressure[m_currentPressureWrite],		// output: sorted pressure
			m_dParticleHash,				// input: sorted grid hashes
			m_dParticleIndex,				// input: sorted particle indices
			m_dPos[m_currentPosRead],			// input: sorted position array
			m_dVel[m_currentVelRead],			// input: sorted velocity array
			m_dInfo[m_currentInfoRead],			// input: sorted info array
			m_dBoundElement[m_currentBoundElementRead],	// input: sorted boundary elements
			m_dGradGamma[m_currentGradGammaRead],		// input: sorted gradient gamma
			m_dVertices[m_currentVerticesRead],		// input: sorted vertices
			m_dPressure[m_currentPressureRead],		// input: sorted pressure
			m_dNewNumParticles,				// output: number of active particles
			m_numParticles,
			m_nGridCells,
			m_dInversedParticleIndex);

	uint activeParticles;
	CUDA_SAFE_CALL(cudaMemcpy(&activeParticles, m_dNewNumParticles,
			sizeof(uint), cudaMemcpyDeviceToHost));
	if (activeParticles != m_numParticles) {
		printf("particles: %d => %d\n", m_numParticles, activeParticles);
		m_timingInfo.numParticles = m_numParticles = activeParticles;
	}


	std::swap(m_currentPosRead, m_currentPosWrite);
	std::swap(m_currentVelRead, m_currentVelWrite);
	std::swap(m_currentInfoRead, m_currentInfoWrite);
	std::swap(m_currentBoundElementRead, m_currentBoundElementWrite);
	std::swap(m_currentGradGammaRead, m_currentGradGammaWrite);
	std::swap(m_currentVerticesRead, m_currentVerticesWrite);
	std::swap(m_currentPressureRead, m_currentPressureWrite);

	m_timingInfo.numInteractions = 0;
	m_timingInfo.maxNeibs = 0;
	resetneibsinfo();

	// Build the neighbours list
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
			m_simparams->periodicbound);

	getneibsinfo(m_timingInfo);
	if (m_timingInfo.maxNeibs > m_simparams->maxneibsnum) {
		printf("WARNING: current max. neighbors numbers %d greather than MAXNEIBSNUM (%d)\n", m_timingInfo.maxNeibs, m_simparams->maxneibsnum);
		fflush(stdout);
	}

	if (timing) {
		cudaEventRecord(stop_neibslist, 0);
		cudaEventSynchronize(stop_neibslist);
		cudaEventElapsedTime(&m_timingInfo.timeNeibsList, start_neibslist, stop_neibslist);
		m_timingInfo.timeNeibsList *= 1e-3;
		cudaEventDestroy(start_neibslist);
		cudaEventDestroy(stop_neibslist);

		int iter = m_iter/m_simparams->buildneibsfreq + 1;
		m_timingInfo.meanNumInteractions = (m_timingInfo.meanNumInteractions*(iter - 1) + m_timingInfo.numInteractions)/iter;
		m_timingInfo.meanTimeNeibsList = (m_timingInfo.meanTimeNeibsList*(iter - 1) + m_timingInfo.timeNeibsList)/iter;
	}

	m_neiblist_built = true;
}

void
ParticleSystem::initializeGammaAndGradGamma(void)
{
	buildNeibList(false);

	//Copy initial values of density from the sorted array (m_currentVelRead) to the unsorted array (m_currentVelWrite),
	//which will be used to set virtual velocities during initilization of gamma field
	long size = m_numParticles*sizeof(float4);
	void* dest = m_dVel[m_currentVelWrite];
	void* orig = m_dVel[m_currentVelRead];
	CUDA_SAFE_CALL(cudaMemcpy(dest, orig, size, cudaMemcpyDeviceToDevice));

	initGradGamma(	m_dPos[m_currentPosRead],
			m_dPos[m_currentPosWrite],
			m_dVel[m_currentVelWrite],
			m_dInfo[m_currentInfoRead],
			m_dBoundElement[m_currentBoundElementRead],
			m_dGradGamma[m_currentGradGammaWrite],
			m_dNeibsList,
			m_numParticles,
			m_problem->m_deltap,
			m_simparams->slength,
			m_influenceRadius,
			m_simparams->kerneltype,
			m_simparams->periodicbound);

	std::swap(m_currentPosRead, m_currentPosWrite);
	std::swap(m_currentGradGammaRead, m_currentGradGammaWrite);
	std::swap(m_currentVelRead, m_currentVelWrite);

	// Compute virtual displacement
	int itNumber = 200;
	float deltat = 1.0/itNumber;

	for(uint i = 0; i < itNumber; i++) {
		// On every iteration updateGamma() is called twice, while updatePositions() is called only once,
		// since evolution equation for gamma is integrated in time with a second-order time scheme:
		// gamma(n+1) = gamma(n) + 0.5*[gradGam(n) + gradGam(n+1)]*[r(n+1) - r(n)]

		// Update gamma 1st call
		updateGamma(	m_dPos[m_currentPosRead],
				m_dPos[m_currentPosWrite],
				m_dVel[m_currentVelRead],
				m_dInfo[m_currentInfoRead],
				m_dBoundElement[m_currentBoundElementRead],
				m_dGradGamma[m_currentGradGammaRead],
				m_dGradGamma[m_currentGradGammaWrite],
				m_dNeibsList,
				m_numParticles,
				m_simparams->slength,
				m_influenceRadius,
				deltat,
				0,
				m_simparams->kerneltype,
				m_simparams->periodicbound);

		std::swap(m_currentGradGammaRead, m_currentGradGammaWrite);

		// Move the particles
		updatePositions(	m_dPos[m_currentPosRead],
					m_dPos[m_currentPosWrite],
					m_dVel[m_currentVelRead],
					m_dInfo[m_currentInfoRead],
					deltat,
					m_numParticles);

		std::swap(m_currentPosRead, m_currentPosWrite);

		// Build the neighbour list
		buildNeibList(false);

		// Update gamma 2nd call
		updateGamma(	m_dPos[m_currentPosRead],
				m_dPos[m_currentPosWrite],
				m_dVel[m_currentVelRead],
				m_dInfo[m_currentInfoRead],
				m_dBoundElement[m_currentBoundElementRead],
				m_dGradGamma[m_currentGradGammaRead],
				m_dGradGamma[m_currentGradGammaWrite],
				m_dNeibsList,
				m_numParticles,
				m_simparams->slength,
				m_influenceRadius,
				deltat,
				0,
				m_simparams->kerneltype,
				m_simparams->periodicbound);

		std::swap(m_currentGradGammaRead, m_currentGradGammaWrite);

		//DEBUG output
//		getArray(POSITION, false);
//		getArray(INFO, false);
//		getArray(GRADGAMMA, false);
//		std::string fname = m_problem->get_dirname() + "/gradgamma_init.csv";
//		FILE *fp = fopen(fname.c_str(), "a");
//		for (uint index = 0; index < m_numParticles; index++)
//			if(FLUID(m_hInfo[index])) {
//			float4 pos = m_hPos[index];
//			float4 gradGam = m_hGradGamma[index];
//		
//			fprintf(fp, "%f,%f,%f,%f,%f\n", pos.z, gradGam.x, gradGam.y, gradGam.z, gradGam.w);
//			}
//		fclose(fp);
	}
}

void
ParticleSystem::updateValuesAtBoundaryElements(void)
{
	updateBoundValues(	m_dVel[m_currentVelRead],
				m_dPressure[m_currentPressureRead],
				m_dVertices[m_currentVerticesRead],
				m_dInfo[m_currentInfoRead],
				m_numParticles,
				true);
}

void
ParticleSystem::imposeDynamicBoundaryConditions(void)
{
	dynamicBoundConditions(	m_dPos[m_currentPosRead],
				m_dVel[m_currentVelRead],
				m_dPressure[m_currentPressureRead],
				m_dInfo[m_currentInfoRead],
				m_dNeibsList,
				m_numParticles,
				m_simparams->slength,
				m_simparams->kerneltype,
				m_influenceRadius,
				m_simparams->periodicbound);
}

TimingInfo
ParticleSystem::PredcorrTimeStep(bool timing)
{
	// do nothing if the simulation is over
	if (m_problem->finished(m_simTime))
		return m_timingInfo;

	cudaEvent_t start_interactions, stop_interactions;
	cudaEvent_t start_euler, stop_euler;

	if (m_iter % m_simparams->buildneibsfreq == 0) {
		buildNeibList(timing);
	}

	if (m_simparams->shepardfreq > 0 && m_iter > 0 && (m_iter % m_simparams->shepardfreq == 0)) {
		shepard(m_dPos[m_currentPosRead],
				m_dVel[m_currentVelRead],
				m_dVel[m_currentVelWrite],
				m_dInfo[m_currentInfoRead],
				m_dNeibsList,
				m_numParticles,
				m_simparams->slength,
				m_simparams->kerneltype,
				m_influenceRadius,
				m_simparams->periodicbound);

		std::swap(m_currentVelRead, m_currentVelWrite);
	}

	if (m_simparams->mlsfreq > 0 && m_iter > 0 && (m_iter % m_simparams->mlsfreq == 0)) {
		mls(	m_dPos[m_currentPosRead],
				m_dVel[m_currentVelRead],
				m_dVel[m_currentVelWrite],
				m_dInfo[m_currentInfoRead],
				m_dNeibsList,
				m_numParticles,
				m_simparams->slength,
				m_simparams->kerneltype,
				m_influenceRadius,
				m_simparams->periodicbound);

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
	if (m_simparams->mbcallback) {
		float4* hMbData = m_problem->get_mbdata(m_simTime + m_dt/2.0, m_dt/2.0, m_iter == 0);
		if (hMbData)
			setmbdata(hMbData, m_mbDataSize);
		}
	if (m_simparams->gcallback) {
		m_physparams->gravity = m_problem->g_callback(m_simTime);
		setgravity(m_physparams->gravity);
	}

	float3 *cg;
	float3 *trans;
	float *rot;

	// Copying floating bodies centers of gravity for torque computation in forces (needed only at first
	// setp)
	if (m_simparams->numbodies && m_iter == 0) {
		cg = m_problem->get_rigidbodies_cg();
		setforcesrbcg(cg, m_simparams->numbodies);
		seteulerrbcg(cg, m_simparams->numbodies);

//		// Debug
//		for (int i=0; i < m_simparams->numbodies; i++) {
//			printf("Body %d: cg(%g,%g,%g) lastindex: %d\n", i, cg[i].x, cg[i].y, cg[i].z, m_hRbLastIndex[i]);
//			}
//
//		uint rbfirstindex[MAXBODIES];
//		CUDA_SAFE_CALL(cudaMemcpyFromSymbol(rbfirstindex, "d_rbstartindex", m_simparams->numbodies*sizeof(uint)));
//		for (int i=0; i < m_simparams->numbodies; i++) {
//			printf("Body %d: firstindex: %d\n", i, rbfirstindex[i]);
//			}
		}

	if(m_simparams->boundarytype == MF_BOUNDARY)
	{
		//update density and pressure at vertex particles
		dynamicBoundConditions(	m_dPos[m_currentPosRead], 		//pos(n)
					m_dVel[m_currentVelRead], 		//vel(n)
					m_dPressure[m_currentPressureRead],
					m_dInfo[m_currentInfoRead],
					m_dNeibsList,
					m_numParticles,
					m_simparams->slength,
					m_simparams->kerneltype,
					m_influenceRadius,
					m_simparams->periodicbound);

		updateBoundValues(	m_dVel[m_currentVelRead],		//vel(n)
					m_dPressure[m_currentPressureRead],
					m_dVertices[m_currentVerticesRead],
					m_dInfo[m_currentInfoRead],
					m_numParticles,
					false);
	}

	dt1 = forces(   m_dPos[m_currentPosRead],   // pos(n)
					m_dVel[m_currentVelRead],   // vel(n)
					m_dForces,					// f(n)
					m_dGradGamma[m_currentGradGammaRead],
					m_dBoundElement[m_currentBoundElementRead],
					m_dPressure[m_currentPressureRead],
					m_dRbForces,
					m_dRbTorques,
					m_dXsph,
					m_dInfo[m_currentInfoRead],
					m_dNeibsList,
					m_numParticles,
					m_simparams->slength,
					m_dt,
					m_simparams->dtadapt,
					m_simparams->dtadaptfactor,
					m_simparams->xsph,
					m_simparams->kerneltype,
					m_influenceRadius,
					m_simparams->visctype,
					m_physparams->visccoeff,
					m_dCfl,
					m_dCflGamma,
					m_dTempCfl,
					m_numPartsFmax,
					m_dTau,
					m_simparams->periodicbound,
					m_simparams->sph_formulation,
					m_simparams->boundarytype,
					m_simparams->usedem);
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

	if (m_simparams->numbodies) {
		reduceRbForces(m_dRbForces, m_dRbTorques, m_dRbNum, m_hRbLastIndex, m_hRbTotalForce,
						m_hRbTotalTorque, m_simparams->numbodies, m_numBodiesParticles);

		m_problem->rigidbodies_timestep(m_hRbTotalForce, m_hRbTotalTorque, 1, m_dt, cg, trans, rot);
		seteulerrbtrans(trans, m_simparams->numbodies);
		seteulerrbsteprot(rot, m_simparams->numbodies);
	}

	euler(  m_dPos[m_currentPosRead],   // pos(n)
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
			m_simparams->xsph,
			m_simparams->periodicbound);
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

	// euler need the previous center of gravity but forces the new, so we copy to GPU
	// here instead before call to euler
	if (m_simparams->numbodies) {
		setforcesrbcg(cg, m_simparams->numbodies);
		seteulerrbcg(cg, m_simparams->numbodies);
	}

	// setting moving boundaries data if necessary
	if (m_simparams->mbcallback) {
		float4* hMbData = m_problem->get_mbdata(m_simTime + m_dt, m_dt/2.0, m_iter == 0);
		if (hMbData)
			setmbdata(hMbData, m_mbDataSize);
		}
	if (m_simparams->gcallback) {
		m_physparams->gravity = m_problem->g_callback(m_simTime);
		setgravity(m_physparams->gravity);
	}

	if(m_simparams->boundarytype == MF_BOUNDARY)
	{
		//update density and pressure at vertex particles
		dynamicBoundConditions(	m_dPos[m_currentPosWrite], 		//pos(n+1/2)
					m_dVel[m_currentVelWrite], 		//vel(n+1/2)
					m_dPressure[m_currentPressureRead],
					m_dInfo[m_currentInfoRead],
					m_dNeibsList,
					m_numParticles,
					m_simparams->slength,
					m_simparams->kerneltype,
					m_influenceRadius,
					m_simparams->periodicbound);

		updateBoundValues(	m_dVel[m_currentVelWrite],		//vel(n+1/2)
					m_dPressure[m_currentPressureRead],
					m_dVertices[m_currentVerticesRead],
					m_dInfo[m_currentInfoRead],
					m_numParticles,
					false);

		// gamma(n+1/2) = gamma(n) + dt/4*∑[gradGam(n+1/2) + gradGam(n)] * Vel(n+1/2)
		updateGamma(	m_dPos[m_currentPosRead],			//pos(n)
				m_dPos[m_currentPosWrite],			//pos(n+1/2)
				m_dVel[m_currentVelWrite],			//vel(n+1/2)
				m_dInfo[m_currentInfoRead],
				m_dBoundElement[m_currentBoundElementRead],
				m_dGradGamma[m_currentGradGammaRead],		//gamma(n) {input}
				m_dGradGamma[m_currentGradGammaWrite],		//gamma(n+1/2) {output}
				m_dNeibsList,
				m_numParticles,
				m_simparams->slength,
				m_influenceRadius,
				m_dt,
				1,
				m_simparams->kerneltype,
				m_simparams->periodicbound);

		std::swap(m_currentGradGammaRead, m_currentGradGammaWrite);
		// At this point:
		// m_dGradGamma[m_currentGradGammaRead] = gradgamma(n+1/2)
		// m_dGradGamma[m_currentGradGammaWrite] = gradgamma(n)

	}

	dt2 = forces(   m_dPos[m_currentPosWrite],  // pos(n+1/2)
					m_dVel[m_currentVelWrite],  // vel(n+1/2)
					m_dForces,					// f(n+1/2)
					m_dGradGamma[m_currentGradGammaRead],
					m_dBoundElement[m_currentBoundElementRead],
					m_dPressure[m_currentPressureRead],
					m_dRbForces,
					m_dRbTorques,
					m_dXsph,
					m_dInfo[m_currentInfoRead],
					m_dNeibsList,
					m_numParticles,
					m_simparams->slength,
					m_dt,
					m_simparams->dtadapt,
					m_simparams->dtadaptfactor,
					m_simparams->xsph,
					m_simparams->kerneltype,
					m_influenceRadius,
					m_simparams->visctype,
					m_physparams->visccoeff,
					m_dCfl,
					m_dCflGamma,
					m_dTempCfl,
					m_numPartsFmax,
					m_dTau,
					m_simparams->periodicbound,
					m_simparams->sph_formulation,
					m_simparams->boundarytype,
					m_simparams->usedem);
	// At this point forces = f(pos(n+1/2), vel(n+1/2))

	if (m_simparams->numbodies) {
		reduceRbForces(m_dRbForces, m_dRbTorques, m_dRbNum, m_hRbLastIndex, m_hRbTotalForce,
						m_hRbTotalTorque, m_simparams->numbodies, m_numBodiesParticles);

		m_problem->rigidbodies_timestep(m_hRbTotalForce, m_hRbTotalTorque, 2, m_dt, cg, trans, rot);
		seteulerrbtrans(trans, m_simparams->numbodies);
		seteulerrbsteprot(rot, m_simparams->numbodies);
	}

	euler(  m_dPos[m_currentPosRead],   // pos(n)
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
			m_simparams->xsph,
			m_simparams->periodicbound);
	// At this point:
	//  m_dPos[m_currentPosRead] = pos(n)
	//  m_dVel[m_currentVelRead] =  vel(n)
	//  m_dForces = f(n+1/2)
	//  m_dPos[m_currentPosWrite] = pos(n+1) = pos(n) + velc(n+1/2)*dt
	//  m_dVel[m_currentVelWrite] =  vel(n+1) = vel(n) + f(n+1/2)*dt

	if(m_simparams->boundarytype == MF_BOUNDARY)
	{
		// gamma(n+1) = gamma(n+1/2) + dt/4*∑[gradGam(n+1) + gradGam(n+1/2)] * Vel(n+1)
		updateGamma(	m_dPos[m_currentPosRead],			//pos(n)
				m_dPos[m_currentPosWrite],			//pos(n+1)
				m_dVel[m_currentVelWrite],			//vel(n+1)
				m_dInfo[m_currentInfoRead],
				m_dBoundElement[m_currentBoundElementRead],
				m_dGradGamma[m_currentGradGammaRead],		//gamma(n+1/2) {input}
				m_dGradGamma[m_currentGradGammaWrite],		//gamma(n+1) {output}
				m_dNeibsList,
				m_numParticles,
				m_simparams->slength,
				m_influenceRadius,
				m_dt,
				1,
				m_simparams->kerneltype,
				m_simparams->periodicbound);

		std::swap(m_currentGradGammaRead, m_currentGradGammaWrite);
		// At this point:
		// m_dGradGamma[m_currentGradGammaRead] = gradgamma(n+1)
		// m_dGradGamma[m_currentGradGammaWrite] = gradgamma(n+1/2))
	}

	// euler need the previous center of gravity but forces the new, so we copy to GPU
	// here instead before call to euler
	if (m_simparams->numbodies) {
		setforcesrbcg(cg, m_simparams->numbodies);
		seteulerrbcg(cg, m_simparams->numbodies);
	}

	//Calculate values at probe particles
	if(true)
	{
		calcProbe(	m_dPos[m_currentPosWrite],
				m_dVel[m_currentVelWrite],
				m_dPressure[m_currentPressureRead],
				m_dInfo[m_currentInfoRead],
				m_dNeibsList,
				m_numParticles,
				m_simparams->slength,
				m_simparams->kerneltype,
				m_influenceRadius,
				m_simparams->periodicbound);
	}

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
	if (m_simparams->dtadapt) {
		m_dt = min(dt1, dt2);   // next time step value
		if (!m_dt) {
			throw DtZeroException(m_simTime, m_dt);
		} else if (m_dt < FLT_EPSILON) {
			fprintf(stderr, "[%g] WARNING! new timestep %g too small!\n",
					m_simTime, m_dt);
		}
	}
	fflush(stderr);

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
	getArray(POSITION, false);
	getArray(NEIBSLIST, false);
	std::string fname;
	fname = m_problem->get_dirname() + "/neibs.txt";
	FILE *fp = fopen(fname.c_str(),"w");
	for (uint index = 0; index < m_numParticles; index++) {
		float3 pos;
		pos.x = m_hPos[index].x;
		pos.y = m_hPos[index].y;
		pos.z = m_hPos[index].z;

		// TODO: fix it for neib index interleave
		for(uint i = index*m_simparams->maxneibsnum; i < index*m_simparams->maxneibsnum + m_simparams->maxneibsnum; i++) {
			uint neib_index = m_hNeibsList[i];

			if (neib_index == 0xffffffff) break;

			int3 periodic = make_int3(0);
			if (m_simparams->periodicbound) {
				if (neib_index & WARPXPLUS)
					periodic.x = 1;
				else if (neib_index & WARPXMINUS)
					periodic.x = -1;
				if (neib_index & WARPYPLUS)
					periodic.y = 1;
				else if (neib_index & WARPYMINUS)
					periodic.y = -1;
				if (neib_index & WARPZPLUS)
					periodic.z = 1;
				else if (neib_index & WARPZMINUS)
					periodic.z = -1;

				neib_index &= NOWARP;
			}

			float3 neib_pos;
			neib_pos.x = m_hPos[neib_index].x;
			neib_pos.y = m_hPos[neib_index].y;
			neib_pos.z = m_hPos[neib_index].z;

			float3 relPos;
			relPos.x = pos.x - neib_pos.x;
			relPos.y = pos.y - neib_pos.y;
			relPos.z = pos.z - neib_pos.z;
			float3 relPos2;
			relPos2 = relPos + periodic*m_physparams->dispvect;

			fprintf(fp, "%d\t%f\t%f\t%f\t", index, pos.x, pos.y, pos.z);
			fprintf(fp, "%d\t%f\t%f\t%f\t", neib_index, relPos.x, relPos.y, relPos.z);
			fprintf(fp, "%d\t%d\t%d\t%f\t%f\t%f\n", periodic.x, periodic.y, periodic.z, relPos2.x, relPos2.y, relPos2.z);
			}
		}
	fclose(fp);
}


void
ParticleSystem::savehash()
{
	getArray(POSITION, false);
	getArray(HASH, false);
	std::string fname;
	fname = m_problem->get_dirname() + "/hash.txt";
	FILE *fp = fopen(fname.c_str(),"w");
	for (uint index = 0; index < m_numParticles; index++) {
		float3 pos;
		pos.x = m_hPos[index].x;
		pos.y = m_hPos[index].y;
		pos.z = m_hPos[index].z;
		hashKey hash = m_hParticleHash[index];
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
	getArray(POSITION, false);
	getArray(PARTINDEX, false);
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
	getArray(POSITION, false);
	getArray(PARTINDEX, false);
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
	getArray(POSITION, false);
	getArray(CELLSTART, false);
	getArray(CELLEND, false);
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
	getArray(NORMALS, false);
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
ParticleSystem::savegradgamma()
{
	getArray(POSITION, false);
	getArray(GRADGAMMA, false);
	getArray(INFO, false);
	std::string fname;
	//fname = m_problem->get_dirname() + "/gradgamma.csv";
	std::stringstream niter;
	niter << m_iter;
	fname = m_problem->get_dirname() + "/gradgamma" + niter.str() + ".dat";
	FILE *fp = fopen(fname.c_str(), "w");

	for (uint index = 0; index < m_numParticles; index++) {
		float4 pos = m_hPos[index];
		float4 gradGam = m_hGradGamma[index];
		
		if(pos.x < 0.1 && pos.y < 0.1 && pos.z < 0.1 && FLUID(m_hInfo[index]))
		fprintf(fp, "%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\n", index, m_hInfo[index].z, pos.x, pos.y, pos.z, gradGam.x, gradGam.y, gradGam.z, gradGam.w, PART_TYPE(m_hInfo[index]));
	}
	fclose(fp);
}

void
ParticleSystem::saveboundelem()
{
	getArray(BOUNDELEMENT, false);
	std::string fname;
	fname = m_problem->get_dirname() + "/boundelements.csv";
	FILE *fp = fopen(fname.c_str(), "w");
	for (uint index = 0; index < m_numParticles; index++) {
		float4 bElm = m_hBoundElement[index];
		
		fprintf(fp, "%d,%f,%f,%f,%f\n", index, bElm.x, bElm.y, bElm.z, bElm.w);
	}
	fclose(fp);
}

void
ParticleSystem::saveVelocity()
{
	getArray(POSITION, false);
	getArray(VELOCITY, false);
	getArray(INFO, false);
	std::string fname;
	//fname = m_problem->get_dirname() + "/velocity.csv";
	std::stringstream niter;
	niter << m_iter;
	fname = m_problem->get_dirname() + "/velocity" + niter.str() + ".csv";
	FILE *fp = fopen(fname.c_str(), "w");
	for (uint index = 0; index < m_numParticles; index++) {
		float4 vel = m_hVel[index];
		float4 pos = m_hPos[index];

		fprintf(fp, "%d,%d,%f,%f,%f,%f,%f,%d\n", index, m_hInfo[index].z, pos.z, vel.x, vel.y, vel.z, vel.w, PART_TYPE(m_hInfo[index]));
	}
	fclose(fp);
}

void
ParticleSystem::savepressure()
{
	getArray(PRESSURE, false);
	getArray(INFO, false);
	std::string fname;
	std::stringstream niter;
	niter << m_iter;
	fname = m_problem->get_dirname() + "/pressure" + niter.str() + ".csv";
	FILE *fp = fopen(fname.c_str(), "w");
	for (uint index = 0; index < m_numParticles; index++) {

		fprintf(fp, "%d,%d,%f\n", index, PART_TYPE(m_hInfo[index]), m_hPressure[index]);
	}
	fclose(fp);
}

void
ParticleSystem::saveForces()
{
	getArray(INFO, false);
	getArray(FORCE, false);
	std::string fname;
	//fname = m_problem->get_dirname() + "/forces.csv";
	std::stringstream niter;
	niter << m_iter;
	fname = m_problem->get_dirname() + "/forces" + niter.str() + ".csv";
	FILE *fp = fopen(fname.c_str(), "w");
	for (uint index = 0; index < m_numParticles; index++)
	if(FLUID(m_hInfo[index]))
	{
		float4 forces = m_hForces[index];
		float4 pos = m_hPos[index];

		fprintf(fp, "%d,%d,%f,%f,%f,%f,%d\n", index, m_hInfo[index].z, forces.x, forces.y, forces.z, forces.w, PART_TYPE(m_hInfo[index]));
	}
	fclose(fp);
}

void
ParticleSystem::saveprobedata()
{
	getArray(PRESSURE, false);
	getArray(POSITION, false);
	getArray(INFO, false);
	std::string fname;
	float H[4][50];
	for (int i=0; i<4; i++)
	for (int j=0; j<50; j++)
		H[i][j] = -1.0f;
	float P[8] = {-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f};

	for (uint index = 0; index < m_numParticles; index++) {
		if (PROBE(m_hInfo[index])) {
			const float x = m_hPos[index].x;
			const float z = m_hPos[index].z;

			//Pressure probes
			if(x>2.3 && x<2.55) {
				if (z<0.04) {//P1
					P[0] = m_hPressure[index];
				}
				else if (z<0.08) {//P2
					P[1] = m_hPressure[index];
				}
				else if (z<0.12) {//P3
					P[2] = m_hPressure[index];
				}
				else if (z<0.16) {//P4
					P[3] = m_hPressure[index];
				}
				else if (x<2.42) {//P5
					P[4] = m_hPressure[index];
				}
				else if (x<2.46) {//P6
					P[5] = m_hPressure[index];
				}
				else if (x<2.50) {//P7
					P[6] = m_hPressure[index];
				}
				else { //P8
					P[7] = m_hPressure[index];
				}
			}
			//H1
			else if(x>2.55) {
				H[0][int(z/0.0199)] = m_hPos[index].w;
			}
			//H2
			else if(x>2.0) {
				H[1][int(z/0.0199)] = m_hPos[index].w;
			}
			//H3
			else if(x>1.5) {
				H[2][int(z/0.0199)] = m_hPos[index].w;
			}
			//H4
			else {
				H[3][int(z/0.0199)] = m_hPos[index].w;
			}
		}
	}
	for (int i=0; i<4; i++) {
		for (int j=0; j<49; j++)
		{
			H[i][0] += H[i][j] + H[i][j+1];
		}
		H[i][0] *= 0.02*0.5;
	}

	fname = m_problem->get_dirname() + "/probes_p.dat";
	FILE *fp = fopen(fname.c_str(), "a");
		fprintf(fp, "%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n", this->getTime(), P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7]);
	fclose(fp);

	fname = m_problem->get_dirname() + "/probes_h.dat";
	fp = fopen(fname.c_str(), "a");
		fprintf(fp, "%g\t%g\t%g\t%g\t%g\n", this->getTime(), H[0][0], H[1][0], H[2][0], H[3][0]);
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
	for (int i = 0; i < m_simparams->numbodies; i++) {
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

// WaveGage 
void
ParticleSystem::writeWaveGage()
{
	float uplimitx, downlimitx, uplimity, downlimity;

	int num =  m_simparams->gage.size();

	for (int i = 0 ; i < num; ++i) {
		uplimitx = m_simparams->gage[i].x+2*m_simparams->slength;
		downlimitx = m_simparams->gage[i].x-2*m_simparams->slength;
		uplimity = m_simparams->gage[i].y+2*m_simparams->slength;
		downlimity = m_simparams->gage[i].y-2*m_simparams->slength;
		int kcheck = 0;
		m_simparams->gage[i].z = 0;

		for (uint index = 0; index < m_numParticles; index++) {
			if (SURFACE_PARTICLE(m_hInfo[index])) {
				float4 pos = m_hPos[index];

				//Taking height average between neighbouring surface particles 
				if ((pos.x > downlimitx) && (pos.x < uplimitx) && (pos.y > downlimity) && (pos.y < uplimity)){
					kcheck ++;
					m_simparams->gage[i].z += pos.z;
				}
			} //if PART_FLAG
		} //For loop over particles
		m_simparams->gage[i].z = m_simparams->gage[i].z/kcheck;
	} // For loop over WaveGages

	//Write WaveGage information on one text file
	m_writer->write_WaveGage(m_simTime,m_simparams->gage);

	//Writing the result on a VTK files	
	// TODO this should be factored out in a Writer routine, and possibly made optional
	static int fnum = 0;
	stringstream ss;

	ss.width(5);
	ss.fill('0');
	ss << m_problem->get_dirname() << "/data/WaveGage_" << fnum << ".vtu";

	fnum++;

	FILE *fp = fopen(ss.str().c_str(),"w");

	// Header
	fprintf(fp,"<?xml version=\"1.0\"?>\r\n");
	fprintf(fp,"<VTKFile type= \"UnstructuredGrid\"  version= \"0.1\"  byte_order= \"BigEndian\">\r\n");
	fprintf(fp," <UnstructuredGrid>\r\n");
	fprintf(fp,"  <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\r\n", num, num);	
	
	//Writing Position	
	fprintf(fp,"   <Points>\r\n");
	fprintf(fp,"	<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\r\n");
	for (int i=0; i <  num; i++)
		fprintf(fp,"%f\t%f\t%f\t",m_simparams->gage[i].x, m_simparams->gage[i].y, m_simparams->gage[i].z);
	fprintf(fp,"\r\n");
	fprintf(fp,"	</DataArray>\r\n");
	fprintf(fp,"   </Points>\r\n");
	
	// Cells data
	fprintf(fp,"   <Cells>\r\n");
	fprintf(fp,"	<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\r\n");
	for (int i = 0; i < num; i++)
		fprintf(fp,"%d\t", i);
	fprintf(fp,"\r\n");
	fprintf(fp,"	</DataArray>\r\n");
	fprintf(fp,"\r\n");
	
	fprintf(fp,"	<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\r\n");
	for (int i = 0; i < num; i++)
		fprintf(fp,"%d\t", i + 1);
	fprintf(fp,"\r\n");
	fprintf(fp,"	</DataArray>\r\n");
	
	fprintf(fp,"\r\n");
	fprintf(fp,"	<DataArray type=\"Int32\" Name=\"types\" format=\"ascii\">\r\n");
	for (int i = 0; i < num; i++)
		fprintf(fp,"%d\t", 1);
	fprintf(fp,"\r\n");
	fprintf(fp,"	</DataArray>\r\n");
	
	fprintf(fp,"   </Cells>\r\n");
	
	fprintf(fp,"  </Piece>\r\n");
	fprintf(fp," </UnstructuredGrid>\r\n");
	fprintf(fp,"</VTKFile>");	
	fclose(fp);	
}


/****************************************************************************************************/
