#ifndef DTATFILEPARSER_H
#define	DTATFILEPARSER_H

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "vector_math.h"

class DataFileParser {
	public:
		DataFileParser() {};

		DataFileParser(const char* fname) {
		    std::ifstream file;
		    file.open(fname);
		    if (!file.is_open())
		    	throw std::runtime_error("Cannot open data file\n");

		    std::string line;
		    while (std::getline(file, line)) {
		    	// Skip commented line
		    	if (line[0] == '#' || line.size() <= 4)
		    		continue;
		    	std::vector<std::string> tokens;
		    	Tokenize(line, tokens);

		    	std::cout << line.size() << "\t" << line << "\t" << tokens[0] << "\t" << tokens[1] << "\n";

		    	if (tokens.size() != 2)
			    	throw std::runtime_error("Syntax error (more or less than 2 fields per line)\n");
		    	sVar[tokens[0]] = tokens[1];
		    }
		};

		~DataFileParser() {};

		void saveVariables(const char* fname) {
		    std::ofstream file;
		    file.open(fname);
		    if (!file.is_open())
		    	throw std::runtime_error("Cannot open output file\n");
		    outputVariables(file);
		    file.close();
		};

		void printVariables() {
			outputVariables(std::cout);
		};

		bool has(const std::string& key) {
			return (sVar.find(key) != sVar.end());
		};

		int getI(const std::string& key) {
			if (has(key)) {
				std::istringstream buffer(sVar[key]);
				int res;
				buffer >> res;
				return res;
			}
			else
		    	throw std::runtime_error("Unknown key\n");
		};

		double getD(const std::string& key) {
			if (has(key)) {
				std::istringstream buffer(sVar[key]);
				double res;
				buffer >> res;
				return res;
			}
			else
		    	throw std::runtime_error("Unknown key\n");
		};

		float getF(const std::string& key) {
			return (float) getD(key);
		};

		double3 getD3(const std::string& key) {
			if (has(key)) {
		    	std::vector<std::string> tokens;
		    	Tokenize(sVar[key], tokens, ",");
		    	if (tokens.size() != 3)
			    	throw std::runtime_error("Wrong number (!=3) of doubles for getD3\n");
		    	double3 res;
		    	double *resptr = &(res.x);
		    	for (int i = 0; i < 3; i++) {
					std::istringstream buffer(tokens[i]);
					buffer >> resptr[i];
		    	}
		    	return res;
			}
			else
		    	throw runtime_error("Unknown key\n");
		};

		float3 getF3(const std::string& key) {
			return make_float3(getD3(key));
		};

		string getS(const std::string& key) {
			if (has(key))
				return sVar[key];
			else
				throw std::runtime_error("Unknown key\n");
		};

	private:
		static void Tokenize(const string& str, vector<string>& tokens, const string& delimiters = " ") {
			// Skip delimiters at beginning.
			string::size_type lastPos = str.find_first_not_of(delimiters, 0);
			// Find first "non-delimiter".
			string::size_type pos = str.find_first_of(delimiters, lastPos);
			while (string::npos != pos || string::npos != lastPos) {
				// Found a token, add it to the vector.
				tokens.push_back(str.substr(lastPos, pos - lastPos));
				// Skip delimiters.  Note the "not_of"
				lastPos = str.find_first_not_of(delimiters, pos);
				// Find next "non-delimiter"
				pos = str.find_first_of(delimiters, lastPos);
			}
		};

		void outputVariables(std::ostream & file) {
		    map<string, string>::iterator iter;
		    for(iter = sVar.begin(); iter != sVar.end(); iter++ )
		        file << iter->first << " " << iter->second << "\n";
		};

		map<string, string> sVar;
};

#endif
