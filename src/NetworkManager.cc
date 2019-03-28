/*
 * NetworkManager.cc
 *
 *  Created on: Jan 11, 2013
 *      Author: rustico
 */

/*! \file
 * NetworkManager class implementation
 */

#include "mpi_select.opt"

#if USE_MPI
#include <mpi.h>
#else
#define MPI_MAX_PROCESSOR_NAME 256
#define NO_MPI_ERR throw runtime_error("MPI support not compiled in")
#endif

#include "NetworkManager.h"
// for GlobalData::RANK()
#include <GlobalData.h>

#if USE_MPI
static MPI_Request* m_requestsList;
#endif

using namespace std;

// Uncomment the following to define DBG_PRINTF and enable printing the details of every call (uint and buffer).
// Useful to check the correspondence among messages without compiling in debug mode
//#define DBG_PRINTF

NetworkManager::NetworkManager() {
	// TODO Auto-generated constructor stub
	processor_name = new char[MPI_MAX_PROCESSOR_NAME];
	processor_name[0] = '\0';
	// cout << MPI_MAX_PROCESSOR_NAME; // 256

	world_size = 0; // 1 process = single node. 0 is reserved for "uninitialized"
	process_rank = -1; // -1 until initialization is done
	processor_name_len = 0;

	// MPIRequests for asynchronous calls
	m_numRequests = 0;
	m_requestsCounter = 0;
#if USE_MPI
	m_requestsList = NULL;
#endif
}

NetworkManager::~NetworkManager() {
#if USE_MPI
	// TODO Auto-generated destructor stub
	// TODO: finalize if not done yet
	// MPIRequests for asynchronous calls
	if (m_requestsList)
		free(m_requestsList);
#endif
	delete[] processor_name;
	processor_name = NULL;
	processor_name_len = 0;
}

void NetworkManager::setNumRequests(uint _numRequests)
{
	m_numRequests = _numRequests;
#if USE_MPI
	m_requestsList = (MPI_Request*)realloc(m_requestsList, m_numRequests * sizeof(MPI_Request));
#endif
}

void NetworkManager::initNetwork() {
	// initialize the MPI environment
	// MPI_Init(NULL, NULL);
#if USE_MPI
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
#else
	world_size = 1;
	process_rank = 0;
#endif
}

void NetworkManager::finalizeNetwork(int ret) {
	// finalize the MPI environment
#if USE_MPI
	if (ret)
		MPI_Abort(MPI_COMM_WORLD, ret);
	else
		MPI_Finalize();
#endif
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

//! Combine a source and destination global device index into an MPI message tag
static inline
unsigned int
exchange_tag(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx)
{
	return (((unsigned int)src_globalDevIdx) << GLOBAL_DEVICE_BITS) | dst_globalDevIdx;
}

//! Combine a source and destination global device index and a buffer id into an MPI message tag
static inline
unsigned int
async_exchange_tag(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, uint bid)
{
	unsigned int base_tag = exchange_tag(src_globalDevIdx, dst_globalDevIdx);
	return (bid << (2*GLOBAL_DEVICE_BITS)) | base_tag;
}

// print world size,process name and rank
void NetworkManager::printInfo()
{
	printf("[Network] rank %u (%u/%u), host %s\n", process_rank, process_rank + 1, world_size, processor_name);
}

void NetworkManager::sendUint(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int *datum)
{
#if USE_MPI
	unsigned int tag = exchange_tag(src_globalDevIdx, dst_globalDevIdx);

#ifdef DBG_PRINTF
	printf("  ---- MPI UINT src %u dst %u cnt %u tag %u\n", src_globalDevIdx, dst_globalDevIdx, 4, tag);
#endif

	int mpi_err = MPI_Send(datum, 1, MPI_INT, GlobalData::RANK(dst_globalDevIdx), tag, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Send returned error %d\n", mpi_err);
#else
	NO_MPI_ERR;
#endif
}

void NetworkManager::receiveUint(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int *datum)
{
#if USE_MPI
	unsigned int tag = exchange_tag(src_globalDevIdx, dst_globalDevIdx);

#ifdef DBG_PRINTF
	printf("  ---- MPI UINT src %u dst %u cnt %u tag %u\n", src_globalDevIdx, dst_globalDevIdx, 4, tag);
#endif

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
#else
	NO_MPI_ERR;
#endif
}

void NetworkManager::sendBuffer(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, void *src_data)
{
#if USE_MPI
	unsigned int tag = exchange_tag(src_globalDevIdx, dst_globalDevIdx);

#ifdef DBG_PRINTF
	printf("  ---- MPI BUFFER src %u dst %u cnt %u tag %u\n", src_globalDevIdx, dst_globalDevIdx, count, tag);
#endif

	int mpi_err = MPI_Send(src_data, count, MPI_BYTE, GlobalData::RANK(dst_globalDevIdx), tag, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Send returned error %d\n", mpi_err);
#else
	NO_MPI_ERR;
#endif
}

void NetworkManager::receiveBuffer(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, void *dst_data)
{
#if USE_MPI
	unsigned int tag = exchange_tag(src_globalDevIdx, dst_globalDevIdx);

#ifdef DBG_PRINTF
	printf("  ---- MPI BUFFER src %u dst %u cnt %u tag %u\n", src_globalDevIdx, dst_globalDevIdx, count, tag);
#endif

	MPI_Status status;
	int mpi_err = MPI_Recv(dst_data, count, MPI_BYTE, GlobalData::RANK(src_globalDevIdx), tag, MPI_COMM_WORLD, &status);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Recv returned error %d\n", mpi_err);
	int actual_count;

	mpi_err = MPI_Get_count(&status, MPI_BYTE, &actual_count);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Get_count returned error %d\n", mpi_err);
	else
	if (actual_count != count)
		printf("WARNING: MPI_Get_count returned %d (bytes), expected %u\n", actual_count, count);
#else
	NO_MPI_ERR;
#endif
}

void NetworkManager::sendBufferAsync(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, void *src_data, uint bid)
{
#if USE_MPI
	unsigned int tag = async_exchange_tag(src_globalDevIdx, dst_globalDevIdx, bid);
	int mpi_err = 0;

	#ifdef DBG_PRINTF
	printf("  ---- MPI BUFFER ASYNC src %u dst %u cnt %u tag %u\n", src_globalDevIdx, dst_globalDevIdx, count, tag);
	#endif

	if (m_requestsCounter == (m_numRequests-1))
		printf("WARNING: NetworkManager: %u was set as max number of requests, ignoring SEND!\n",
			m_numRequests);
	else
		mpi_err = MPI_Isend(src_data, count, MPI_BYTE, GlobalData::RANK(dst_globalDevIdx), tag, MPI_COMM_WORLD,
			&m_requestsList[m_requestsCounter++]);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_ISend returned error %d\n", mpi_err);
#else
	NO_MPI_ERR;
#endif
}

void NetworkManager::receiveBufferAsync(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, void *dst_data, uint bid)
{
#if USE_MPI
	unsigned int tag = async_exchange_tag(src_globalDevIdx, dst_globalDevIdx, bid);
	int mpi_err = 0;

	#ifdef DBG_PRINTF
	printf("  ---- MPI BUFFER ASYNC src %u dst %u cnt %u tag %u\n", src_globalDevIdx, dst_globalDevIdx, count, tag);
	#endif

	if (m_requestsCounter == (m_numRequests-1))
		printf("WARNING: NetworkManager: %u was set as max number of requests, ignoring RECV!\n",
			m_numRequests);
	else
		mpi_err = MPI_Irecv(dst_data, count, MPI_BYTE, GlobalData::RANK(src_globalDevIdx), tag, MPI_COMM_WORLD,
		&m_requestsList[m_requestsCounter++]);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_IRecv returned error %d\n", mpi_err);

	/* mpi_err = MPI_Get_count(&status, MPI_BYTE, &actual_count);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Get_count returned error %d\n", mpi_err);
	else
		if (actual_count != count)
			printf("WARNING: MPI_Get_count returned %d (bytes), expected %u\n", actual_count, count); */
#else
	NO_MPI_ERR;
#endif
}

void NetworkManager::waitAsyncTransfers()
{
#if USE_MPI
	if (m_requestsCounter > 0)
		MPI_Waitall(m_requestsCounter, m_requestsList, MPI_STATUSES_IGNORE);

	// if one needs to check statuses one by one:
	/*
	for (uint i=0; i < m_requestsCounter; i++) {
		MPI_Status status;
		int actual_count;

		MPI_Wait(&(m_requestsList[i]), &status);
		// or:
		// MPI_Wait(&(m_requestsList[i]), MPI_STATUS_IGNORE);

		int mpi_err = MPI_Wait(&m_requestsList[i], &status);
		MPI_Get_count(&status, MPI_BYTE, &actual_count);
		// here we can read status.MPI_TAG and status.MPI_SOURCE for checking / debugging, but beware:
		// status can be reset on the sender's side if successful
	}
	*/
	m_requestsCounter = 0;
#else
	NO_MPI_ERR;
#endif
}


#if 0
void NetworkManager::sendUints(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, uint *src_data)
{
	unsigned int tag = exchange_tag(src_globalDevIdx, dst_globalDevIdx);

	int mpi_err = MPI_Send(src_data, count, MPI_UNSIGNED, GlobalData::RANK(dst_globalDevIdx), tag, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Send returned error %d\n", mpi_err);
}

void NetworkManager::receiveUints(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, uint *dst_data)
{
	unsigned int tag = exchange_tag(src_globalDevIdx, dst_globalDevIdx);

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
	unsigned int tag = exchange_tag(src_globalDevIdx, dst_globalDevIdx);

	int mpi_err = MPI_Send(src_data, count, MPI_FLOAT, GlobalData::RANK(dst_globalDevIdx), tag, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Send returned error %d\n", mpi_err);
}

void NetworkManager::receiveFloats(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, float *dst_data)
{
	unsigned int tag = exchange_tag(src_globalDevIdx, dst_globalDevIdx);

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
	unsigned int tag = exchange_tag(src_globalDevIdx, dst_globalDevIdx);

	int mpi_err = MPI_Send(src_data, count, MPI_SHORT, GlobalData::RANK(dst_globalDevIdx), tag, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Send returned error %d\n", mpi_err);
}

void NetworkManager::receiveShorts(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, unsigned short *dst_data)
{
	unsigned int tag = exchange_tag(src_globalDevIdx, dst_globalDevIdx);

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

void NetworkManager::networkFloatReduction(float *buffer, const unsigned int bufferElements, ReductionType rtype)
{
#if USE_MPI
	MPI_Op _operator;
	switch (rtype) {
		case MIN_REDUCTION:
			_operator = MPI_MIN;
			break;
		case MAX_REDUCTION:
			_operator = MPI_MAX;
			break;
		case SUM_REDUCTION:
			_operator = MPI_SUM;
			break;
		default:
			_operator = MPI_SUM;
			printf("WARNING: Wrong operator in networkFloatReduction specified. Defaulting to SUM_REDUCTION.\n");
	}

	int mpi_err = MPI_Allreduce(MPI_IN_PLACE, buffer, bufferElements, MPI_FLOAT, _operator, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Allreduce returned error %d\n", mpi_err);
#else
	NO_MPI_ERR;
#endif
}

void NetworkManager::networkIntReduction(int *buffer, const unsigned int bufferElements, ReductionType rtype)
{
#if USE_MPI
	MPI_Op _operator;
	switch (rtype) {
		case MIN_REDUCTION:
			_operator = MPI_MIN;
			break;
		case MAX_REDUCTION:
			_operator = MPI_MAX;
			break;
		case SUM_REDUCTION:
			_operator = MPI_SUM;
			break;
		default:
			_operator = MPI_SUM;
			printf("WARNING: Wrong operator in networkFloatReduction specified. Defaulting to SUM_REDUCTION.\n");
	}

	int mpi_err = MPI_Allreduce(MPI_IN_PLACE, buffer, bufferElements, MPI_INTEGER, _operator, MPI_COMM_WORLD);

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Allreduce returned error %d\n", mpi_err);
#else
	NO_MPI_ERR;
#endif
}

void NetworkManager::networkBoolReduction(bool *buffer, const unsigned int bufferElements)
{
#if USE_MPI
	// we need a int buffer since MPI doesn't have a bool type
	// Note: We need to use MPI_INT and not MPI_CHAR as MPI_CHAR cannot be applied to the MPI_MAX operator
	int ibuffer[bufferElements];
	for (uint i=0; i<bufferElements; i++)
		ibuffer[i] = buffer[i] ? 1 : 0;

	int mpi_err = MPI_Allreduce(MPI_IN_PLACE, &ibuffer, bufferElements, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

	for (uint i=0; i<bufferElements; i++)
		buffer[i] = ibuffer[i] > 0;

	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Allreduce returned error %d\n", mpi_err);
#else
	NO_MPI_ERR;
#endif
}

// send one int, gather the int from all nodes (allgather)
void NetworkManager::allGatherUints(unsigned int *datum, unsigned int *recv_buffer)
{
#if USE_MPI
	int mpi_err = MPI_Allgather(datum, 1, MPI_INT, recv_buffer, 1, MPI_INT, MPI_COMM_WORLD);
	if (mpi_err != MPI_SUCCESS)
		printf("WARNING: MPI_Allgather returned error %d\n", mpi_err);
#else
	NO_MPI_ERR;
#endif
}

// network barrier
void NetworkManager::networkBarrier()
{
#if USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
}

#if USE_MPI
// We send the maximum allowed tag value to denote the kill request
static const unsigned int kill_tag = MPI_TAG_UB;
static const unsigned char kill_payload = 0xFFU;
#endif

void NetworkManager::sendKillRequest()
{
#if USE_MPI
	// We want to broadcast the kill request, which is usually handled by the root process,
	// so we would have to send the kill request from the dying process to the root, and then the root
	// would broadcast, but we don't care about being efficient or clean in this case, so let's just
	// do the broadcast ourselves “manually”

	// printf("Sending out MPI kill request ...\n");
	for (int target_rank = 0; target_rank < world_size; ++target_rank) {
		if (target_rank != process_rank) {
			MPI_Request dont_care;
			MPI_Isend(&kill_payload, 1, MPI_CHAR, target_rank, kill_tag, MPI_COMM_WORLD, &dont_care);
			MPI_Request_free(&dont_care);
		}
	}
	// printf("... done. Cya on the other side.\n");
#endif
}

bool NetworkManager::checkKillRequest()
{
#if USE_MPI
	int found = 0;
	MPI_Status status;
	int mpi_err = MPI_Iprobe(MPI_ANY_SOURCE, kill_tag, MPI_COMM_WORLD, &found, &status);
	if (mpi_err == MPI_SUCCESS && found) {
		// recycle found to find the number of bytes of the request.
		mpi_err |= MPI_Get_count(&status, MPI_CHAR, &found);
	}
	// return true if found is exactly 1 and no errors
	return mpi_err == MPI_SUCCESS && found == 1;
#else
	return false;
#endif
}


#ifdef DBG_PRINTF
#undef DBG_PRINTF
#endif
