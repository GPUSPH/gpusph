/*
  Name:		gl_screenshot
  Author:	  eSCHEn
  Contact:	 eschen@gibbering.net
			   http://legion/gibbering.net/evillution/
  Date:		16 Jul 04
  Copyright:   Use it, have fun.  Author retains copyright.
			   If you use this in a project then send me a
			   mail - I'd love to see what you're doing with
			   it :)
			   Released under MPL 1.1, see 'mpl.txt' for
			   details.
  Description: Exports a class 'CScreenshot', which takes
			   a screenshot of an OpenGL context and saves
			   it to a specified targa file.

  Usage of Class:
  ===============

  Methods:
	CounterPosition -> Current position of internal counter.
	SetCounter	  -> Sets the initial position of internal counter.
	SetFileName	 -> Sets the base filename for saving.
	TakeScreenshot  -> Takes the screenshot using internal filename.
	TakeScreenshot  -> Takes the screenshot, <Filename> overrides internal
					   filename.

  Properties:
	AutoIncrement -> Set to true if auto-incrementing of counter is needed,
					 will add auto-numbering to the end of the internal
					 filename.
*/

#ifndef _GL_SCREENSHOT_H_
#define _GL_SCREENSHOT_H_

#include <fstream>
#include <string>
#include <stdlib.h>

using namespace std;

struct CTGAHeader
{
	char				IDCount;
	char				ColorMap;
	char				ImageCode;
	unsigned short		CMOrigin;
	char				CMLength;
	char				CMBitDepth;
	unsigned short		IMXOrigin;
	unsigned short		IMYOrigin;
	unsigned short		IMWidth;
	unsigned short		IMHeight;
	char				IMBitDepth;
	char				IMDescByte;
};


class CScreenshot
{
	public:
	// constructor
	CScreenshot(std::string dirname);
	// destructor
	virtual ~CScreenshot();

	// methods
	int CounterPosition();
	void SetCounter(int CountValue);
	void SetFileName(char* FileName);
	void TakeScreenshot(float t);
	void TakeScreenshot(char* FileName, float t);
	char* LastFilename() {return fLastFileName;};

	// variables
	bool AutoIncrement; // set to false to stop auto-numbering

  private:
	// variables
	const char*	fDefaultFileName;
	int		fFileCount;
	char*	fLastFileName;
	string	m_dirname;
	FILE	*m_timefile;
};
#endif