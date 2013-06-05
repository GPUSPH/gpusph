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
#ifndef H_UDPWRITER_H
#define H_UDPWRITER_H

#include "Writer.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>

using namespace std;
/*
UDP packet writer.
Sends UDP packets containing test point data in realtime.  Default host:port
is 127.0.0.1:8889.  User can override by setting environment variables
UDPWRITER_HOST and/or UDPWRITER_PORT.
*/
#define UDP_PACKET_SIZE 1024*32
class UDPWriter : public Writer
{
public:
	UDPWriter(Problem *problem);
	~UDPWriter();

	void write(uint numParts, const float4 *pos, const float4 *vel,
		const particleinfo *info, const float3 *vort, float t, const bool
        testpoints, const float4 *normals);
protected:
    float3 world_origin, world_size;
    pthread_t heartbeat_thread;
    /** buffer for composing packet */
    char mBuf[UDP_PACKET_SIZE];
    struct sockaddr_in remote_client_address;
    socklen_t remote_client_address_len;
    static void *heartbeat_thread_main(void *user_data);

    /** server address */
    struct sockaddr_in mServerAddress;

    /** socket */
    int mSocket;

    /** port number */
    int mPort;

    /** hostname */
    char mHost[INET6_ADDRSTRLEN];

    int heartbeat_socket_fd;
    struct sockaddr_in client_address;
    socklen_t client_address_len;
};

#endif
