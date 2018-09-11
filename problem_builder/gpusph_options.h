#ifndef _GPUSPH_OPTIONS_H_
#define _GPUSPH_OPTIONS_H_
#include "ini/cpp/INIReader.h"
#include <string>

class GPUSPHOptions {
	public:
		std::string ProblemName;
		void getGeneralOptions(INIReader config);
};
#endif
