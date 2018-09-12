
#ifndef REPACK_ALGO_H_
#define REPACK_ALGO_H_

#include "common_types.h"
#include <string>

class GPUSPH;
struct GlobalData;
class Problem;

/**
  \class Repacking
  Class implementing the auxiliary stages for repacking algorithm
*/
class Repacking
{
public:
	Repacking();
	~Repacking();

	void Init( GPUSPH*, GlobalData*, Problem* );
	bool SetParams();
	bool Start();
	void Stop();

	float TotalKE() const;
	float maxC0() const;

protected:
	std::string GetFileName( bool isHdf ) const;
	bool HdfFileExists( const std::string& ) const;
	bool ParametersEqual( const std::string& ) const;
	void LoadData( const std::string& ) const;
	void SaveData( const std::string& ) const;
	void SaveParams( const std::string& ) const;

private:
	GPUSPH* gpusph;			///< the pointer to GPUSPH solver
	GlobalData* gdata;		///< the pointer to global data of solver
	Problem* problem;		///< the pointer to current problem
	float max_c0;			///< the maximum "numerical" speed of sound
	float dt;				///< the original solver dt
};

#endif
