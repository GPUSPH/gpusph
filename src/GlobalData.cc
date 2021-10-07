/*  Copyright (c) 2013-2019 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

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

/*! \file GlobalData.cc
 * Implementation of GlobalData methods for which we need actual class definitions
 */

#include <cstdio>
#include <sstream>

#include "GlobalData.h"

#include "ProblemCore.h"
#include "NetworkManager.h"

GlobalData::~GlobalData()
{
		delete problem;
		if (networkManager)
			networkManager->finalizeNetwork(ret);
		delete networkManager;
}

//! Cleanup between REPACKING and SIMULATE runs
//! TODO FIXME this rather fragile at the moment,
//! using less naked pointers and proper RAII would help with that.
void GlobalData::cleanup()
{
	printf("Cleaning GlobalData...\n");

	GPUWORKERS.clear();
	threadSynchronizer = NULL;

	s_hRbCgGridPos = NULL;
	s_hRbCgPos = NULL;
	s_hRbTranslations = NULL;
	s_hRbLinearVelocities = NULL;
	s_hRbAngularVelocities = NULL;
	s_hRbRotationMatrices = NULL;

	s_hRbFirstIndex = NULL;
	s_hRbLastIndex = NULL;
	s_hRbTotalForce = NULL;
	s_hRbAppliedForce = NULL;
	s_hRbTotalTorque = NULL;
	s_hRbAppliedTorque = NULL;
	s_hRbDeviceTotalForce = NULL;
	s_hRbDeviceTotalTorque = NULL;

	s_hDeviceMap = NULL;
	s_hPartsPerSliceAlongX = NULL;
	s_hPartsPerSliceAlongY = NULL;
	s_hPartsPerSliceAlongZ = NULL;
	s_dCellEnds = NULL;
	s_dCellStarts = NULL;
	s_dSegmentsStart = NULL;

	allocPolicy = NULL;
	simframework = NULL;
	delete problem;
	problem = NULL;
	totParticles = 0;
	numOpenVertices = 0;
	allocatedParticles = 0;
	nGridCells = 0;
	particlesCreated = false;
	createdParticlesIterations = 0;
	keep_going = true;
	quit_request = false;
	save_request = false;
	iterations = 0;
	maxiter = ULONG_MAX;
	t = 0.0;
	dt = 0.0f;
	lastGlobalPeakFluidBoundaryNeibsNum = 0;
	lastGlobalPeakVertexNeibsNum = 0;
	lastGlobalNumInteractions = 0;
	nextCommand = IDLE;
}

void GlobalData::saveDeviceMapToFile(std::string prefix) const
{
	std::ostringstream oss;
	oss << problem->get_dirname() << "/";
	if (!prefix.empty())
		oss << prefix << "_";
	oss << problem->m_name;
	oss << "_dp" << problem->m_deltap;
	if (mpi_nodes > 1) oss << "_rank" << mpi_rank << "." << mpi_nodes << "." << networkManager->getProcessorName();
	oss << ".csv";
	std::string fname = oss.str();
	FILE *fid = fopen(fname.c_str(), "w");
	fprintf(fid,"X,Y,Z,LINEARIZED,VALUE\n");
	for (uint ix=0; ix < gridSize.x; ix++)
		for (uint iy=0; iy < gridSize.y; iy++)
			for (uint iz=0; iz < gridSize.z; iz++) {
				uint cell_lin_idx = calcGridHashHost(ix, iy, iz);
				fprintf(fid,"%u,%u,%u,%u,%u\n", ix, iy, iz, cell_lin_idx, s_hDeviceMap[cell_lin_idx]);
			}
	fclose(fid);
	printf(" > device map dumped to file %s\n",fname.c_str());
}

void GlobalData::saveCompactDeviceMapToFile(std::string prefix, uint srcDev, uint *compactDeviceMap) const
{
	std::ostringstream oss;
	oss << problem->get_dirname() << "/";
	if (!prefix.empty())
		oss << prefix << "_";
	oss << problem->m_name;
	oss << "_dp" << problem->m_deltap;
	if (devices > 1) oss << "_dev" << srcDev << "." << devices;
	oss << ".csv";
	std::string fname = oss.str();
	FILE *fid = fopen(fname.c_str(), "w");
	fprintf(fid,"X,Y,Z,LINEARIZED,VALUE\n");
	for (uint ix=0; ix < gridSize.x; ix++)
		for (uint iy=0; iy < gridSize.y; iy++)
			for (uint iz=0; iz < gridSize.z; iz++) {
				uint cell_lin_idx = calcGridHashHost(ix, iy, iz);
				fprintf(fid,"%u,%u,%u,%u,%u\n", ix, iy, iz, cell_lin_idx, compactDeviceMap[cell_lin_idx] >> 30);
			}
	fclose(fid);
	printf(" > compact device map dumped to file %s\n",fname.c_str());
}
