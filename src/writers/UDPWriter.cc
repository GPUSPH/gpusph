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

#include <sstream>
#include <unistd.h>
#include <iostream>
#include <stdexcept>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>

#include "UDPWriter.h"
#include "GlobalData.h"
#include "ptp.h"

//#define DEBUG

using namespace std;

void *UDPWriter::heartbeat_thread_main(void *user_data) {
    UDPWriter *w = (UDPWriter*)user_data;

    /* option value */
    int optval = 1;

    /* address */
    struct sockaddr_in my_address;

    /* setup address */
    memset(&my_address, 0, sizeof(my_address));
    my_address.sin_family      = AF_INET;
    my_address.sin_port        = htons(w->mPort);
    if(w->mHost[0] == '\0') {
        // for binding to all local addresses
        my_address.sin_addr.s_addr = INADDR_ANY;
    } else {
        // for binding to a specific address
        inet_aton(w->mHost, &my_address.sin_addr);
    }

    /* create server socket */
    if ((w->mHeartbeatSocketFd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP))
        == -1) {
        perror("socket");
        pthread_exit(NULL);
    }

    /* reuse local address */
    if (setsockopt(w->mHeartbeatSocketFd, SOL_SOCKET, SO_REUSEADDR, &optval,
                  sizeof optval) == -1) {
        perror("setsockopt(SO_REUSEADDR)");
        close(w->mHeartbeatSocketFd);
        pthread_exit(NULL);
    }

#ifdef SET_BUFFER_SIZE
    int udp_buffer_size = sizeof(ptp_packet_t) * 1024;
    if (setsockopt(w->mHeartbeatSocketFd, SOL_SOCKET, SO_SNDBUF,
        &udp_buffer_size, (socklen_t)(sizeof(int))) == -1) {
        perror("setsockopt(SO_SNDBUF)");
    }

    if (setsockopt(w->mHeartbeatSocketFd, SOL_SOCKET, SO_RCVBUF,
        &udp_buffer_size, (socklen_t)(sizeof(int))) == -1) {
        perror("setsockopt(SO_RCVBUF)");
    }
#endif
    /* bind to local address:port */
    if (bind(w->mHeartbeatSocketFd, (struct sockaddr *)&my_address,
        sizeof(my_address)) == -1) {
        perror("bind");
        pthread_exit(NULL);
    }

    /* set non-blocking so we can manage timing */
    fcntl(w->mHeartbeatSocketFd, F_SETFL, O_NONBLOCK);

    /* Loop until application asks us to exit */
    int done = 0;
    time_t last_heartbeat_received = 0;
    time_t now;
    while(!done) {
       struct sockaddr_in from;
       socklen_t fromlen = sizeof(from);

        /* packet variable */
        ptp_heartbeat_packet_t packet;

        /* length of packet received */
        ssize_t packet_length_bytes;

        /* receive a packet */
        packet_length_bytes = recvfrom(w->mHeartbeatSocketFd, &packet,
                                       sizeof(ptp_heartbeat_packet_t),
                                       0, (struct sockaddr*)&from, &fromlen);
        if(packet_length_bytes == 0) {
            /* socket closed */
            done = 1;
        } else if(packet_length_bytes ==-1) {
            if(errno == EAGAIN) {
            } else {
                perror("recvfrom");
            }
        }
        /* did we receive a packet? */
        if (packet_length_bytes == sizeof(ptp_heartbeat_packet_t)) {

            // update internal client address information
            w->mClientAddressLen = fromlen;
            memcpy(&w->mClientAddress, &from, w->mClientAddressLen);
            char str[INET6_ADDRSTRLEN];
            if(inet_ntop(AF_INET,
                &from.sin_addr.s_addr,
                str, INET6_ADDRSTRLEN) == NULL) {
                perror("inet_ntop");
            }
#ifdef DEBUG
            fprintf(stdout, "Received packet from address: %s\n", str);
#endif
            time(&last_heartbeat_received);
        }
        time(&now);
        size_t d = difftime(now, last_heartbeat_received);
        if(d > (PTP_HEARTBEAT_TTL_S * 2)) {
            w->mClientAddressLen = 0;
        }
        usleep(1);
    }

    return(NULL);
}

/* Print pthreads error user-defined and internal error message. */
#define PT_ERR_MSG(str, code) { \
        fprintf(stderr, "%s: %s\n", str, strerror(code)); \
}

UDPWriter::UDPWriter(const GlobalData *_gdata): Writer(_gdata) {
    // if UDPWRITER_HOST or UDPWRITER_PORT environment variables are set,
    // use those values, otherwise defaults
    mPort = PTP_DEFAULT_SERVER_PORT;
    char *p = getenv("UDPWRITER_HOST");
    if(p) {
        sprintf(mHost, "%s", p);
    } else {
        mHost[0] = '\0';
    }
    mWorldOrigin = gdata->problem->get_worldorigin();
    mWorldSize =  gdata->problem->get_worldsize();
    if((p = getenv("UDPWRITER_PORT"))) {
        mPort = atoi(p);
    }
    int err;
    if ((err = pthread_create(&mHeartbeatThread, NULL, heartbeat_thread_main,
        (void*)this))) {
        PT_ERR_MSG("heartbeat pthread_create", err);
    }
    memset(&mClientAddress, 0, sizeof(mClientAddress));
    mClientAddressLen = 0;

    if ((mSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1) {
        perror("socket");
        pthread_exit(NULL);
    }
    int udp_buffer_size = sizeof(ptp_packet_t) * 1024 * 1024;
    if (setsockopt(mSocket, SOL_SOCKET, SO_SNDBUF,
        &udp_buffer_size, (socklen_t)(sizeof(int))) == -1) {
        perror("setsockopt(SO_SNDBUF)");
    }
}

UDPWriter::~UDPWriter() {
    close(mSocket);
}

void
UDPWriter::write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints)
{
	const double4 *pos = buffers.getData<BUFFER_POS_GLOBAL>();
	const float4 *vel = buffers.getData<BUFFER_VEL>();
	const particleinfo *info = buffers.getData<BUFFER_INFO>();

    static short is_initialized = 0;
    static int particles_in_last_packet = 0;
    static uint packets_sent = 0;
    static int packets_per_time_step = 0;
    static ptp_packet_t packet;

    if(!is_initialized) {
        particles_in_last_packet = numParts % PTP_PARTICLES_PER_PACKET;
        double v = numParts / PTP_PARTICLES_PER_PACKET;
        packets_per_time_step = v;
        if(v != 0.0) {
            packets_per_time_step++;
        }
#ifdef DEBUG
        cout << numParts << " particles" << endl;
        cout << PTP_PARTICLES_PER_PACKET << " particles per packet" << endl;
        cout << particles_in_last_packet << " particles in last packet" << endl;
        cout << packets_per_time_step << " packets_per_time_step" << endl;
#endif
        is_initialized = 1;

        int udp_buffer_size = sizeof(ptp_packet_t) * packets_per_time_step;
        if (setsockopt(mSocket, SOL_SOCKET, SO_SNDBUF,
            &udp_buffer_size, (socklen_t)(sizeof(int))) == -1) {
            perror("setsockopt(SO_SNDBUF)");
        }

        // Initialize common packet data
        packet.total_particle_count = numParts;
        packet.world_size[0] = mWorldSize.x;
        packet.world_size[1] = mWorldSize.y;
        packet.world_size[2] = mWorldSize.z;
        packet.world_origin[0] = mWorldOrigin.x;
        packet.world_origin[1] = mWorldOrigin.y;
        packet.world_origin[2] = mWorldOrigin.z;
        packet.model_id = getpid();
    }
    if(mClientAddressLen == 0) {
        return;
    } else {
        char str[INET6_ADDRSTRLEN];
        if(inet_ntop(AF_INET,
            &mClientAddress.sin_addr.s_addr,
            str, INET6_ADDRSTRLEN) == NULL) {
            perror("inet_ntop");
        }
    }

    /* set the outgoing port number */
    mClientAddress.sin_port = htons(PTP_DEFAULT_CLIENT_PORT);

    int total_particles_sent = 0;
	for (int pi = 0; pi < packets_per_time_step; pi++) {
        // Send time stamp
        packet.t = t;

        // How many particles in this packet?
        packet.particle_count = (pi == (packets_per_time_step - 1)) ?
            particles_in_last_packet : PTP_PARTICLES_PER_PACKET;

        // Copy particle data into packet
        for(uint i = 0; i < packet.particle_count; i++) {
            int offset = (pi * PTP_PARTICLES_PER_PACKET) + i;
            packet.data[i].id = offset;
            packet.data[i].particle_type = info[offset].x;
            memcpy(&packet.data[i].position, &pos[offset], sizeof(double4));
            total_particles_sent++;
        }

        // Send it
        if(sendto(mSocket, (void*)&packet, sizeof(ptp_packet_t), 0,
            (const sockaddr*)&mClientAddress, mClientAddressLen) == -1) {
            if(mClientAddressLen == 0) {
                /* client went away */
                break;
            }
            perror("sendto");
        }
        packets_sent++;
        //usleep(10);
	}
#ifdef DEBUG
    cout << "sent " << packets_sent << " total packets, " <<
        total_particles_sent << " particles in last packet" << endl;
#endif
}

