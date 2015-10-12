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

/*
 * ptp.h
 *
 *  Created on: May 26, 2013
 *      Author: kgamiel
 */

#ifndef PTP_H_
#define PTP_H_

#define PTP_VERSION 0
#define PTP_UDP_PACKET_MAX 1472
#define PTP_HEARTBEAT_TTL_S 1
#define PTP_DEFAULT_CLIENT_PORT 50000
#define PTP_DEFAULT_SERVER_PORT 50001
#define PTP_DEFAULT_SERVER_HOST "127.0.0.1"
#define PTP_DEFAULT_CLIENT_HOST "127.0.0.1"

typedef struct __attribute__ ((packed)) {
    unsigned int id;
    double position[4];
    short particle_type;
} ptp_particle_data_t;

#define PTP_PACKET_HEADER_SIZE ((2 * sizeof(unsigned int)) + (7 * sizeof(float)))
#define PTP_PARTICLES_PER_PACKET ((PTP_UDP_PACKET_MAX - PTP_PACKET_HEADER_SIZE) / sizeof(ptp_particle_data_t))

typedef struct __attribute__ ((packed)) {
    unsigned char   version;
    pid_t           model_id;
    unsigned int total_particle_count;
    unsigned int particle_count;
    float t;
    float world_origin[3];
    float world_size[3];
    ptp_particle_data_t data[PTP_PARTICLES_PER_PACKET];
} ptp_packet_t;

typedef struct __attribute__ ((packed)) {
	unsigned int count;
} ptp_heartbeat_packet_t;


#endif /* PTP_H_ */
