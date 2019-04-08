#include "strings_utils.h"

std::string getFileContent (const char* pathToFile) {
	std::ifstream t(pathToFile);
	std::stringstream buffer;
	buffer << t.rdbuf();
	std::string cudaFile = buffer.str();
	return cudaFile;
}

void replaceAll(std::string& str, const std::string& from, const std::string& to) {
  size_t start_pos = 0;
  while((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
  }
}

std::string stringToUpper(std::string strToConvert) {
  std::transform(strToConvert.begin(), strToConvert.end(), strToConvert.begin(), ::toupper);

  return strToConvert;
}

int tomacro( int c ) { return ::isalnum(c) ? c : '_'; }

std::string stringToMacro(std::string strToConvert) {
  std::transform(strToConvert.begin(), strToConvert.end(), strToConvert.begin(), tomacro);

  return strToConvert;
}

// Check if the string is the name like <base_name>_<index>
bool isIndexedSection( const std::string& theStr, std::string& theBaseName,
    int& theIndex )
{
  bool isIndex = false;
  theBaseName = theStr;
  theIndex = -1;
  std::string::const_iterator aCur = theStr.end() - 1;
  while( aCur >= theStr.begin() && *aCur != '_' )
  {
    isIndex = isdigit( *aCur );
    if ( !isIndex )
      break;
    aCur--;
  }
  if ( isIndex )
  {
    theIndex = atoi( &( *( aCur + 1 ) ) );
    theBaseName = theStr.substr( 0, aCur - theStr.begin() );
  }
  return isIndex;
}

// Find an indent before the given substring.
std::string getIndent( const std::string& theStr, const std::string& theSubstr )
{
  // Find indent
  size_t aPos = theStr.find( theSubstr );
  if ( aPos == std::string::npos )
    return "";
  size_t aLineStart = theStr.rfind( '\n', aPos ) + 1;
  std::string anIndent = "";
  // TODO: check for non-empty characters in the found indent substring
  if ( aPos > aLineStart )
    anIndent = theStr.substr( aLineStart, aPos - aLineStart );
  return anIndent;
}
