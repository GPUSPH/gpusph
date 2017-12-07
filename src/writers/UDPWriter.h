/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

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

/*
UDP packet writer.
*/
#define UDP_PACKET_SIZE 1024*32
class UDPWriter : public Writer
{
public:
	UDPWriter(const GlobalData *_gdata);
	~UDPWriter();

	virtual void write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints);

protected:
    double3     mWorldOrigin,
                mWorldSize;
    pthread_t   mHeartbeatThread;

    /** buffer for composing packet */
    char        mBuf[UDP_PACKET_SIZE];
    struct      sockaddr_in mRemoteClientAddress;
    socklen_t   mRemoteClientAddressLen;

    static void *heartbeat_thread_main(void *user_data);

    /** server address */
    struct sockaddr_in mServerAddress;

    /** socket */
    int mSocket;

    /** port number */
    int mPort;

    /** hostname */
    char mHost[INET6_ADDRSTRLEN];

    int         mHeartbeatSocketFd;
    struct sockaddr_in  mClientAddress;
    socklen_t           mClientAddressLen;
};

#endif
