/*
 * ptp.h
 *
 *  Created on: May 26, 2013
 *      Author: kgamiel
 */

#ifndef PTP_H_
#define PTP_H_

/* in preliminary testing, 4 gives good performance */
#define PTP_PARTICLES_PER_PACKET 4
#define PTP_HEARTBEAT_TTL_S 5
#define PTP_DEFAULT_CLIENT_PORT 50000
#define PTP_DEFAULT_SERVER_PORT 50001
#define PTP_DEFAULT_SERVER_HOST "127.0.0.1"
#define PTP_DEFAULT_CLIENT_HOST "127.0.0.1"

typedef struct __attribute__ ((packed)) {
    unsigned int id;
    unsigned char flag;
    float position[4];
} ptp_particle_data_t;

typedef struct __attribute__ ((packed)) {
    unsigned int total_particle_count;
    unsigned int particle_count;
    float t;
    ptp_particle_data_t data[PTP_PARTICLES_PER_PACKET];
} ptp_packet_t;

typedef struct __attribute__ ((packed)) {
	unsigned int count;
} ptp_heartbeat_packet_t;

#endif /* PTP_H_ */
