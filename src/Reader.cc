#include <limits.h> // UINT_MAX
#include <stdexcept>
#include <fstream>

#include "Reader.h"

Reader::Reader(void) {
	filename = "";
	npart = UINT_MAX;
	buf = NULL;
}

Reader::~Reader(void) {
	empty();
}

void
Reader::empty()
{
	if(buf != NULL){
		delete [] buf;
		buf = NULL;
	}
}

void
Reader::reset()
{
	empty();
	filename = "";
	npart = UINT_MAX;
}

void
Reader::setFilename(std::string const& fn)
{
	// reset npart
	npart = UINT_MAX;
	// copy filename
	filename = fn;
	// check whether file exists
	std::ifstream f(filename.c_str());
	if(!f.good())
		throw std::invalid_argument(std::string("could not open ") + fn);
	f.close();
}
