#! /bin/bash -f

g++ -std=gnu++11 -lrt -lm problem_builder/generate_gpusph_sources.cpp problem_builder/ini/cpp/INIReader.cpp problem_builder/ini/ini.c problem_builder/utils/strings_utils.cpp problem_builder/gpusph_options.cpp problem_builder/cuda_file.cpp problem_builder/cmd_line_parser.cpp problem_builder/params_file.cpp -o build_problem
