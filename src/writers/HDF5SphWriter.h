
#ifndef _HDF5SPHWRITER_H
#define _HDF5SPHWRITER_H

#include <HDF5SphReader.h>

struct GlobalData;

class HDF5SphWriter : public HDF5SphReader
{
public:
	HDF5SphWriter( GlobalData* );
	~HDF5SphWriter();
  
	void write( const char* );
	ReadParticles* fillData( int& npart ) const;
	
private:
	GlobalData* gdata;
};

#endif
