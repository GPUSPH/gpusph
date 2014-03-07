/* The following defines constitute an abstraction of the cartesian coordinates to easily change
 * the order they are used in cell linearization in the whole GPUSPH. This enables fine tuning
 * of the linearization function for performance improvements. In particular, MULTI_DEVICE
 * simulations will benefit of it when the major split axis is COORD3: this means that all the
 * particles in an edging slice (orthogonal to COORD3 axis) will be consecutive in memory and
 * thus eligible for a single burst transfer.
 * Cells with consecutive COORD1 are consecutive in their linearized index. */

#define COORD1	x
#define COORD2	y
#define COORD3	z
