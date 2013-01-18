/*
 * NetworkManager.cc
 *
 *  Created on: Jan 11, 2013
 *      Author: rustico
 */

#include "NetworkManager.h"
#include <mpi.h>

NetworkManager::NetworkManager() {
	// TODO Auto-generated constructor stub
	processor_name = new char[MPI_MAX_PROCESSOR_NAME];
	// std::cout << MPI_MAX_PROCESSOR_NAME; // 256
}

NetworkManager::~NetworkManager() {
	// TODO Auto-generated destructor stub
}

void NetworkManager::initNetwork() {
	// initialize the MPI environment
	MPI_Init(NULL, NULL);

	// get the global number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// get the rank of self
	MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

	// get the name of the processor
	MPI_Get_processor_name(processor_name, &processor_name_len);
}

void NetworkManager::finalizeNetwork() {
	// finalize the MPI environment
	MPI_Finalize();
}

int NetworkManager::getWorldSize() {
	return world_size;
}

int NetworkManager::getProcessRank() {
	return process_rank;
}

char* NetworkManager::getProcessorName() {
	return processor_name;
}

