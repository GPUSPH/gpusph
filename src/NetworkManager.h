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
	// plus: methods to translate global<->local dev id?
};

#endif /* NETWORKMANAGER_H_ */
