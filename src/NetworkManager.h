/*
 * NetworkManager.h
 *
 *  Created on: Jan 11, 2013
 *      Author: rustico
 */

#ifndef NETWORKMANAGER_H_
#define NETWORKMANAGER_H_

enum ReductionType
{
	MIN_REDUCTION,
	SUM_REDUCTION
};

class NetworkManager {
private:
	int world_size;
	int process_rank;
	// TODO: port to String if ever used
	char *processor_name;
	int processor_name_len;
public:
	NetworkManager();
	~NetworkManager();
	void initNetwork();
	void finalizeNetwork();
	int getWorldSize();
	int getProcessRank();
	char* getProcessorName();
	// print world size,process name and rank
	void printInfo();
	// methods to exchange data
	void sendUint(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int *datum);
	void receiveUint(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int *datum);
	void sendUints(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, unsigned int *src_data);
	void receiveUints(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, unsigned int *dst_data);
	void sendFloats(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, float *src_data);
	void receiveFloats(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, float *dst_data);
	void sendShorts(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, unsigned short *src_data);
	void receiveShorts(unsigned char src_globalDevIdx, unsigned char dst_globalDevIdx, unsigned int count, unsigned short *dst_data);
	// network reduction on float buffer across the network
	void networkFloatReduction(float *buffer, unsigned int bufferElements, ReductionType rtype);
	// send one int, gather the int from all nodes (allgather)
	void allGatherUints(unsigned int *datum, unsigned int *recv_buffer);
	// synchronization barrier among all the nodes of the network
	void networkBarrier();
};

#endif /* NETWORKMANAGER_H_ */
