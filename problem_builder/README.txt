This code automatically generates a user problem from an ini configuration file.

To compile, type:
g++ -lrt -lm generate_gpusph_sources.cpp ini/cpp/INIReader.cpp ini/ini.c utils/strings_utils.cpp gpusph_options.cpp header_file.cpp cuda_file.cpp -o generate_gpusph_sources
