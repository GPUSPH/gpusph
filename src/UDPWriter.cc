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
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <errno.h>
#include <netdb.h>
#include "UDPWriter.h"
#include "Problem.h"
#include "ptp.h"
#include <netinet/in.h>

using namespace std;

void *UDPWriter::heartbeat_thread_main(void *user_data) {
    UDPWriter *w = (UDPWriter*)user_data;

    /* option value */
    int optval = 1;

    /* address */
    struct sockaddr_in my_address;

    /* return values */
    int err;

    /* port as string */
    char port_as_string[32];

    /* setup address */
    memset(&my_address, 0, sizeof(my_address));
    my_address.sin_family      = AF_INET;
    my_address.sin_port        = htons(w->mPort);
    if(w->mHost[0] == '\0') {
        // for binding to all local addresses
        my_address.sin_addr.s_addr = INADDR_ANY;
        cout << "Binding to all local addresses" << endl;
    } else {
        // for binding to a specific address
        inet_aton(w->mHost, &my_address.sin_addr);
        cout << "Binding to " << w->mHost << endl;
    }

    /* create server socket */
    if ((w->heartbeat_socket_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP))
        == -1) {
        perror("socket");
        pthread_exit(NULL);
    }

    /* reuse local address */
    if (setsockopt(w->heartbeat_socket_fd, SOL_SOCKET, SO_REUSEADDR, &optval,
                  sizeof optval) == -1) {
        perror("setsockopt(SO_REUSEADDR)");
        close(w->heartbeat_socket_fd);
        pthread_exit(NULL);
    }

#ifdef SO_REUSEPORT
    /* reuse local port (if available) */
    if (setsockopt(w->heartbeat_socket_fd, SOL_SOCKET, SO_REUSEPORT, &optval,
                  sizeof optval) == -1) {
        perror("setsockopt(SO_REUSEPORT)");
        close(w->heartbeat_socket_fd);
        pthread_exit(NULL);
    }
#endif
#ifdef SET_BUFFER_SIZE
    int udp_buffer_size = sizeof(ptp_packet_t) * 1024;
    if (setsockopt(w->heartbeat_socket_fd, SOL_SOCKET, SO_SNDBUF,
        &udp_buffer_size, (socklen_t)(sizeof(int))) == -1) {
        perror("setsockopt(SO_SNDBUF)");
    }

    if (setsockopt(w->heartbeat_socket_fd, SOL_SOCKET, SO_RCVBUF,
        &udp_buffer_size, (socklen_t)(sizeof(int))) == -1) {
        perror("setsockopt(SO_RCVBUF)");
    }
#endif
    /* bind to local address:port */
    if (bind(w->heartbeat_socket_fd, (struct sockaddr *)&my_address,
        sizeof(my_address)) == -1) {
        perror("bind");
        pthread_exit(NULL);
    }

    /* set non-blocking so we can manage timing */
    fcntl(w->heartbeat_socket_fd, F_SETFL, O_NONBLOCK);

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
        packet_length_bytes = recvfrom(w->heartbeat_socket_fd, &packet,
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
            w->client_address_len = fromlen;
            memcpy(&w->client_address, &from, w->client_address_len);
            char str[INET6_ADDRSTRLEN];
            if(inet_ntop(AF_INET,
                &from.sin_addr.s_addr,
                str, INET6_ADDRSTRLEN) == NULL) {
                perror("inet_ntop");
            }
            fprintf(stdout, "Received packet from address: %s\n", str);
            time(&last_heartbeat_received);
        } 
        time(&now);
        size_t d = difftime(now, last_heartbeat_received);
        if(d > (PTP_HEARTBEAT_TTL_S * 2)) {
            w->client_address_len = 0;
        }
        usleep(1);
    }

    return(NULL);
}

/* Print pthreads error user-defined and internal error message. */
#define PT_ERR_MSG(str, code) { \
        fprintf(stderr, "%s: %s\n", str, strerror(code)); \
}

UDPWriter::UDPWriter(Problem *problem) : Writer(problem) {
    // if UDPWRITER_HOST or UDPWRITER_PORT environment variables are set,
    // use those values, otherwise defaults
    mPort = PTP_DEFAULT_SERVER_PORT;
    char *p = getenv("UDPWRITER_HOST");
    if(p) {
        sprintf(mHost, "%s", p);
    } else {
        mHost[0] = '\0';
    }
    world_origin = problem->get_worldorigin();
    world_size =  problem->get_worldsize();
    if((p = getenv("UDPWRITER_PORT"))) {
        mPort = atoi(p);
    }
    int err;
    if ((err = pthread_create(&heartbeat_thread, NULL, heartbeat_thread_main,
        (void*)this))) {
        PT_ERR_MSG("heartbeat pthread_create", err);
    }
    memset(&client_address, 0, sizeof(client_address));
    client_address_len = 0;

    if ((mSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1) {
        perror("socket");
        pthread_exit(NULL);
    }
}

UDPWriter::~UDPWriter() {
    close(mSocket);
}

void UDPWriter::write(uint numParts, const float4 *pos, const float4 *vel,
    const particleinfo *info, const float3 *vort, float t,
    const bool testpoints, const float4 *normals) {
 
    static short is_initialized = 0;
    int particles_in_last_packet = 0;
    static uint packets_sent = 0;
    static int packets_per_time_step = 0;
    static ptp_packet_t packet;

    if(!is_initialized) {
        particles_in_last_packet = numParts % PTP_PARTICLES_PER_PACKET;
        packets_per_time_step = ceil(numParts / PTP_PARTICLES_PER_PACKET);
        cout << numParts << " particles" << endl;
        cout << PTP_PARTICLES_PER_PACKET << " particles per packet" << endl;
        cout << particles_in_last_packet << " particles in last packet" << endl;
        cout << packets_per_time_step << " packets_per_time_step" << endl;
        is_initialized = 1;
    
        // Initialize common packet data
        packet.total_particle_count = numParts;
        packet.world_size[0] = world_size.x;
        packet.world_size[1] = world_size.y;
        packet.world_size[2] = world_size.z;
        packet.world_origin[0] = world_origin.x;
        packet.world_origin[1] = world_origin.y;
        packet.world_origin[2] = world_origin.z;
    }
    if(client_address_len == 0) {
        cout << "No client available, no data sent." << endl;
        return;
    } else {
        char str[INET6_ADDRSTRLEN];
        if(inet_ntop(AF_INET,
            &client_address.sin_addr.s_addr,
            str, INET6_ADDRSTRLEN) == NULL) {
            perror("inet_ntop");
        }
        cout << "Sending data to client " << str << endl;
    }

    /* set the outgoing port number */
    client_address.sin_port = htons(PTP_DEFAULT_CLIENT_PORT);

	for (int pi = 0; pi < packets_per_time_step; pi++) {
        // Send time stamp
        packet.t = t;

        // How many particles in this packet?
        packet.particle_count = (pi == (packets_per_time_step - 1)) ?
            particles_in_last_packet : PTP_PARTICLES_PER_PACKET;

        // Copy particle data into packet
        for(int i = 0; i < packet.particle_count; i++) {
            int offset = (pi * PTP_PARTICLES_PER_PACKET) + i;
            packet.data[i].id = offset;
            packet.data[i].particle_type = info[i].x;
            if(packet.data[i].particle_type != 48) {
                printf("type=%i\n", packet.data[i].particle_type);
            }
            memcpy(&packet.data[i].position, &pos[offset], sizeof(float4));
        }

        // Send it
        if(sendto(mSocket, (void*)&packet, sizeof(ptp_packet_t), 0,
            (const sockaddr*)&client_address, client_address_len) == -1) {
            if(client_address_len == 0) {
                /* client went away */
                break;
            }
            perror("sendto");
        }
        packets_sent++;
        usleep(10);
	}
    cout << "sent " << packets_sent << " packets" << endl;
}
