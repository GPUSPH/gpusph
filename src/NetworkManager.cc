/*
 * NetworkManager.cc
 *
 *  Created on: Jan 11, 2013
 *      Author: rustico
 */

#include "NetworkManager.h"
// for GlobalData::RANK()
#include <GlobalData.h>
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
	// MPI_Init(NULL, NULL);
	int result;
	MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &result);
	if (result < MPI_THREAD_MULTIPLE) {
	    printf("NetworkManager: no complete thread safety, current level: %d\n", result);
	    // MPI_Abort(MPI_COMM_WORLD, 1);
	}

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
	printf("[Network] rank %u (%u/%u), host %s\n", process_rank, process_rank + 1, world_size, processor_name);
}

void NetworkManager::sendUint(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int *datum)
{
	unsigned int tag = ((unsigned int)src_globalDevIdx << 8) | dst_globalDevIdx;

	int mpi_err = MPI_Send(datum, 1, MPI_INT, GlobalData::RANK(dst_globalDevIdx), tag, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Send returned error %d\n", mpi_err);
}

void NetworkManager::receiveUint(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int *datum)
{
	unsigned int tag = ((unsigned int)src_globalDevIdx << 8) | dst_globalDevIdx;

	MPI_Status status;
	int mpi_err = MPI_Recv(datum, 1, MPI_INT, GlobalData::RANK(src_globalDevIdx), tag, MPI_COMM_WORLD, &status);

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

void NetworkManager::sendBuffer(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, void *src_data)
{
	unsigned int tag = ((unsigned int)src_globalDevIdx << 8) | dst_globalDevIdx;

	int mpi_err = MPI_Send(src_data, count, MPI_BYTE, GlobalData::RANK(dst_globalDevIdx), tag, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Send returned error %d\n", mpi_err);
}

void NetworkManager::receiveBuffer(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, void *dst_data)
{
	unsigned int tag = ((unsigned int)src_globalDevIdx << 8) | dst_globalDevIdx;

	MPI_Status status;
	int mpi_err = MPI_Recv(dst_data, count, MPI_BYTE, GlobalData::RANK(src_globalDevIdx), tag, MPI_COMM_WORLD, &status);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Recv returned error %d\n", mpi_err);
	int actual_count;

	mpi_err = MPI_Get_count(&status, MPI_UNSIGNED, &actual_count);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Get_count returned error %d\n", mpi_err);
	else
	if (actual_count != count)
		printf("WARNING: MPI_Get_count returned %d (bytes), expected %u\n", actual_count, count);
}


#if 0
void NetworkManager::sendUints(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, uint *src_data)
{
	unsigned int tag = ((unsigned int)src_globalDevIdx << 8) | dst_globalDevIdx;

	int mpi_err = MPI_Send(src_data, count, MPI_UNSIGNED, GlobalData::RANK(dst_globalDevIdx), tag, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Send returned error %d\n", mpi_err);
}

void NetworkManager::receiveUints(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, uint *dst_data)
{
	unsigned int tag = ((unsigned int)src_globalDevIdx << 8) | dst_globalDevIdx;

	MPI_Status status;
	int mpi_err = MPI_Recv(dst_data, count, MPI_UNSIGNED, GlobalData::RANK(src_globalDevIdx), tag, MPI_COMM_WORLD, &status);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Recv returned error %d\n", mpi_err);
	int actual_count;

	mpi_err = MPI_Get_count(&status, MPI_UNSIGNED, &actual_count);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Get_count returned error %d\n", mpi_err);
	else
	if (actual_count != count)
		printf("WARNING: MPI_Get_count returned %d (uints), expected %u\n", actual_count, count);
}

void NetworkManager::sendFloats(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, float *src_data)
{
	unsigned int tag = ((unsigned int)src_globalDevIdx << 8) | dst_globalDevIdx;

	int mpi_err = MPI_Send(src_data, count, MPI_FLOAT, GlobalData::RANK(dst_globalDevIdx), tag, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Send returned error %d\n", mpi_err);
}

void NetworkManager::receiveFloats(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, float *dst_data)
{
	unsigned int tag = ((unsigned int)src_globalDevIdx << 8) | dst_globalDevIdx;

	MPI_Status status;
	int mpi_err = MPI_Recv(dst_data, count, MPI_FLOAT, GlobalData::RANK(src_globalDevIdx), tag, MPI_COMM_WORLD, &status);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Recv returned error %d\n", mpi_err);
	int actual_count;

	mpi_err = MPI_Get_count(&status, MPI_FLOAT, &actual_count);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Get_count returned error %d\n", mpi_err);
	else
	if (actual_count != count)
		printf("WARNING: MPI_Get_count returned %d (floats), expected %u\n", actual_count, count);
}

void NetworkManager::sendShorts(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, unsigned short *src_data)
{
	unsigned int tag = ((unsigned int)src_globalDevIdx << 8) | dst_globalDevIdx;

	int mpi_err = MPI_Send(src_data, count, MPI_SHORT, GlobalData::RANK(dst_globalDevIdx), tag, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Send returned error %d\n", mpi_err);
}

void NetworkManager::receiveShorts(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, unsigned short *dst_data)
{
	unsigned int tag = ((unsigned int)src_globalDevIdx << 8) | dst_globalDevIdx;

	MPI_Status status;
	int mpi_err = MPI_Recv(dst_data, count, MPI_SHORT, GlobalData::RANK(src_globalDevIdx), tag, MPI_COMM_WORLD, &status);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Recv returned error %d\n", mpi_err);
	int actual_count;

	mpi_err = MPI_Get_count(&status, MPI_SHORT, &actual_count);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Get_count returned error %d\n", mpi_err);
	else
	if (actual_count != count)
		printf("WARNING: MPI_Get_count returned %d (shorts), expected %u\n", actual_count, count);
}
#endif

void NetworkManager::networkFloatReduction(float *buffer, unsigned int bufferElements, ReductionType rtype)
{
	float previous_value = *buffer;
	MPI_Op _operator = (rtype == MIN_REDUCTION ? MPI_MIN : MPI_SUM);

	int mpi_err = MPI_Allreduce(&previous_value, buffer, bufferElements, MPI_FLOAT, _operator, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Allreduce returned error %d\n", mpi_err);
}

// send one int, gather the int from all nodes (allgather)
void NetworkManager::allGatherUints(unsigned int *datum, unsigned int *recv_buffer)
{
	int mpi_err = MPI_Allgather(datum, 1, MPI_INT, recv_buffer, 1, MPI_INT, MPI_COMM_WORLD);
	if (mpi_err != MPI_SUCCESS)
			printf("WARNING: MPI_Allgather returned error %d\n", mpi_err);
}

// network barrier
void NetworkManager::networkBarrier()
{
	MPI_Barrier(MPI_COMM_WORLD);
}
