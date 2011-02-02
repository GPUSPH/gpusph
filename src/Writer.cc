#include "Writer.h"
#include <sstream>
#include <stdexcept>

/**
 *  Default Constructor; makes sure the file output format starts at PART_00000
 */
Writer::Writer(const Problem *problem)
  : m_FileCounter(0), m_problem(problem)
{
	m_dirname = problem->get_dirname() + "/data";
	mkdir(m_dirname.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
}

Writer::~Writer()
{
	// hi
}

string
Writer::next_filenum()
{
	stringstream ss;

	if (m_FileCounter >= MAX_FILES) {
		stringstream ss;
		ss << "too many files created (> " << MAX_FILES;
		throw runtime_error(ss.str());
	}
	ss.width(FNUM_WIDTH);
	ss.fill('0');
	ss << m_FileCounter;

	m_FileCounter++;
	return ss.str();
}

