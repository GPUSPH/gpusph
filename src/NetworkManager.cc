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
	processor_name[0] = '\0';
	// std::cout << MPI_MAX_PROCESSOR_NAME; // 256

	world_size = 0; // 1 process = single node. 0 is reserved for "uninitialized"
	process_rank = -1; // -1 until initialization is done
	processor_name_len = 0;
}

NetworkManager::~NetworkManager() {
	// TODO Auto-generated destructor stub
	// TODO: finalize if not done yet
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

// print world size,process name and rank
void NetworkManager::printInfo()
{
	printf("[Network] rank %u/%u, processor name %s\n", process_rank, world_size, processor_name);
}

void NetworkManager::sendUint(unsigned char dst_rank, unsigned int *datum)
{
	int mpi_err = MPI_Send(datum, 1, MPI_INT, dst_rank, MPI_ANY_TAG, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Send returned error %d\n", mpi_err);
}

void NetworkManager::receiveUint(unsigned char src_rank, unsigned int *datum)
{
	MPI_Status status;
	int mpi_err = MPI_Recv(datum, 1, MPI_INT, src_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Recv returned error %d\n", mpi_err);
	int actual_count;

	mpi_err = MPI_Get_count(&status, MPI_INT, &actual_count);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Get_count returned error %d\n", mpi_err);
	else
	if (actual_count != 1)
		printf("WARNING: MPI_Get_count returned %d (expected 1)\n", actual_count);
}

void NetworkManager::sendFloats(unsigned char dst_rank, unsigned int count, float *src_data)
{
	int mpi_err = MPI_Send(src_data, count, MPI_FLOAT, dst_rank, MPI_ANY_TAG, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Send returned error %d\n", mpi_err);
}

void NetworkManager::receiveFloats(unsigned char src_rank, unsigned int count, float *dst_data)
{
	MPI_Status status;
	int mpi_err = MPI_Recv(dst_data, count, MPI_FLOAT, src_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Recv returned error %d\n", mpi_err);
	int actual_count;

	mpi_err = MPI_Get_count(&status, MPI_INT, &actual_count);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Get_count returned error %d\n", mpi_err);
	else
	if (actual_count != 1)
		printf("WARNING: MPI_Get_count returned %d (expected 1)\n", actual_count);
}

void NetworkManager::sendShorts(unsigned char dst_rank, unsigned int count, unsigned short *src_data)
{
	int mpi_err = MPI_Send(src_data, count, MPI_SHORT, dst_rank, MPI_ANY_TAG, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Send returned error %d\n", mpi_err);
}

void NetworkManager::receiveShorts(unsigned char src_rank, unsigned int count, unsigned short *dst_data)
{
	MPI_Status status;
	int mpi_err = MPI_Recv(dst_data, count, MPI_SHORT, src_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Recv returned error %d\n", mpi_err);
	int actual_count;

	mpi_err = MPI_Get_count(&status, MPI_INT, &actual_count);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Get_count returned error %d\n", mpi_err);
	else
	if (actual_count != 1)
		printf("WARNING: MPI_Get_count returned %d (expected 1)\n", actual_count);
}

void NetworkManager::networkFloatReduction(float *datum)
{
	float previous_value = *datum;
	int mpi_err = MPI_Allreduce(&previous_value, datum, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Allreduce returned error %d\n", mpi_err);
}
