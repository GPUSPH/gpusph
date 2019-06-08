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
/*
 * NetworkManager.h
 *
 *  Created on: Jan 11, 2013
 *      Author: rustico
 */

/*! \file
 * NetworkManager class and related defines
 */

#ifndef NETWORKMANAGER_H_
#define NETWORKMANAGER_H_

typedef unsigned int uint;

enum ReductionType
{
	MIN_REDUCTION,
	MAX_REDUCTION,
	SUM_REDUCTION
};

class NetworkManager {
private:
	int world_size;
	int process_rank;
	// TODO: port to String if ever used
	char *processor_name;
	int processor_name_len;

	uint m_numRequests;
	uint m_requestsCounter;
public:
	NetworkManager();
	~NetworkManager();
	void initNetwork();
	//! Finalize the MPI network connections, with the given return value
	/*! If ret !=, MPI will be aborted, otherwise it will be finalized
	 */
	void finalizeNetwork(int ret);
	int getWorldSize();
	int getProcessRank();
	char* getProcessorName();
	// print world size,process name and rank
	void printInfo();
	// methods to exchange data
	void sendUint(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int *datum);
	void receiveUint(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int *datum);
	void sendBuffer(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, void *src_data);
	void receiveBuffer(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, void *src_data);
	void setNumRequests(uint _numRequests);
	void sendBufferAsync(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, void *src_data, uint bid);
	void receiveBufferAsync(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, void *src_data, uint bid);
	void waitAsyncTransfers();
#if 0
	void sendUints(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, unsigned int *src_data);
	void receiveUints(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, unsigned int *dst_data);
	void sendFloats(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, float *src_data);
	void receiveFloats(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, float *dst_data);
	void sendShorts(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, unsigned short *src_data);
	void receiveShorts(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, unsigned short *dst_data);
#endif
	// network reduction on bool buffer across the network
	void networkBoolReduction(bool *buffer, const unsigned int bufferElements);
	// network reduction on int buffer across the network
	void networkIntReduction(int *buffer, const unsigned int bufferElements, ReductionType rtype);
	// network reduction on float buffer across the network
	void networkFloatReduction(float *buffer, const unsigned int bufferElements, ReductionType rtype);
	// send one int, gather the int from all nodes (allgather)
	void allGatherUints(unsigned int *datum, unsigned int *recv_buffer);
	// synchronization barrier among all the nodes of the network
	void networkBarrier();

	//! Send a message to all other processes letting them know that this process is aborting
	void sendKillRequest();
	//! Check if any process is aborting
	bool checkKillRequest();
};

#endif /* NETWORKMANAGER_H_ */
