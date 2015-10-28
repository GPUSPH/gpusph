#ifndef _VTUREADER_H
#define _VTUREADER_H

#include <string>
#include <iostream>

#include "Reader.h"

class VTUReader : public Reader{
public:
	// returns the number of particles in the vtu file
	int getNParts(void);

	// allocates the buffer and reads the data from the vtu file
	void read(void);
};

#endif
