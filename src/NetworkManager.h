/*
 * NetworkManager.h
 *
 *  Created on: Jan 11, 2013
 *      Author: rustico
 */

#ifndef NETWORKMANAGER_H_
#define NETWORKMANAGER_H_

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
	// methods to exchange data
	void sendUint(unsigned char dst_rank, unsigned int *datum);
	void receiveUint(unsigned char src_rank, unsigned int *datum);
	void sendFloats(unsigned char dst_rank, unsigned int count, float *src_data);
	void receiveFloats(unsigned char src_rank, unsigned int count, float *dst_data);
	void sendShorts(unsigned char dst_rank, unsigned int count, unsigned short *src_data);
	void receiveShorts(unsigned char src_rank, unsigned int count, unsigned short *dst_data);
};

#endif /* NETWORKMANAGER_H_ */
