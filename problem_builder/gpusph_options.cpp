#include "gpusph_options.h"

void GPUSPHOptions::getGeneralOptions(INIReader config) {
	// Get the parameters of the general section
	ProblemName  = config.Get("general", "name", "GenericProblem");
}
