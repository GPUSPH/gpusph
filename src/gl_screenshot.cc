#include "gl_screenshot.h"
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <stdio.h>
#include <string.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>

/* IMPORTANT */
// For an unknown reason, on OsX and only on OsX if, in the same function, you
// use C style (fwrite, fprintf, ...) and C++ style (<<) to write to a file
// the program will crash.

/* constructor */
CScreenshot::CScreenshot(std::string dirname)
{
	// some sensible default parameters
	fDefaultFileName = "image";
	AutoIncrement	= true;
	fFileCount	   = 0;

	m_dirname = dirname + "/images";
	mkdir(m_dirname.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
	std::string time_filename = m_dirname + "/time.txt";
	m_timefile = NULL;
	m_timefile = fopen(time_filename.c_str(), "w");
}

/* destructor */
CScreenshot::~CScreenshot()
{
	if (m_timefile != NULL) {
		fclose(m_timefile);
	}
}

int CScreenshot::CounterPosition()
{
  // get current counter position
  return fFileCount;
}

void CScreenshot::SetCounter(int CountValue)
{
  // copy start position of counter
  fFileCount = CountValue;
}

void CScreenshot::SetFileName(char* FileName)
{
  // copy filename
  fDefaultFileName = FileName;
}

void CScreenshot::TakeScreenshot(float t)
{
  /* grab OpenGL buffer */

  int VPort[4],FSize,PackStore;
  unsigned char *PStore;

  // get viewport dims (x,y,w,h)
  glGetIntegerv(GL_VIEWPORT,VPort);

  // allocate space for framebuffer in rgb format
  FSize = VPort[2]*VPort[3]*3;
  PStore = new unsigned char[FSize];

  // store unpack settings
  glGetIntegerv(GL_PACK_ALIGNMENT, &PackStore);

  // setup unpack settings
  glPixelStorei(GL_PACK_ALIGNMENT, 1);

  // this actually gets the buffer pixels
  glReadPixels(VPort[0],VPort[1],VPort[2],VPort[3],GL_RGB,GL_UNSIGNED_BYTE,PStore);

  // restore unpack settings
  glPixelStorei(GL_PACK_ALIGNMENT, PackStore);

  /* dump data to disk */
  CTGAHeader Header;
  int loop;
  FILE *pFile;
  unsigned char SwapByte;
  char NewFileName[256];

  // setup header
  Header.IDCount	= 0;
  Header.ColorMap   = 0;
  Header.ImageCode  = 2;  // uncompressed rgb image
  Header.CMOrigin   = 0;
  Header.CMLength   = 0;
  Header.CMBitDepth = 0;
  Header.IMXOrigin  = 0;
  Header.IMYOrigin  = 0;
  Header.IMWidth	= VPort[2];
  Header.IMHeight   = VPort[3];
  Header.IMBitDepth = 24; // rgb with no alpha
  Header.IMDescByte = 0;

  // swap red and blue
  for (loop=0;loop<FSize;loop+=3)
  {
	SwapByte	   = PStore[loop];
	PStore[loop]   = PStore[loop+2];
	PStore[loop+2] = SwapByte;
  }

  // setup file name
  if (AutoIncrement)
	sprintf(NewFileName, "%s/%s%05d.tga",m_dirname.c_str(), fDefaultFileName, fFileCount);
  else
	sprintf(NewFileName, "%s/%s.tga",m_dirname.c_str(), fDefaultFileName );

  // copy filename
  fLastFileName = NewFileName;

  // open file
  pFile = fopen(NewFileName,"wb");
  if (pFile==NULL)
  {
	delete[] PStore;
	return;
  }

  // write to disk
  fwrite(&Header, sizeof(CTGAHeader), 1, pFile ); // header
  fwrite(PStore, FSize, 1, pFile ); // image data

  // close file
  fclose(pFile);

  // free memory
	delete[] PStore;

	if (m_timefile != NULL) {
		fprintf(m_timefile, "%s\t%f\n", NewFileName, t);
	}

  // move to next file counter position
  if (AutoIncrement)
	fFileCount++;
}

void CScreenshot::TakeScreenshot(char* FileName, float t)
{
  // copy filename
  fDefaultFileName = FileName;

  // call overloaded screenshot method
  this->TakeScreenshot(t);
}
