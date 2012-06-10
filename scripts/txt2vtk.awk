#!/bin/awk
{
	id[NR-1]=$1
	type[NR-1]=$2
	object[NR-1]=$3
	pos[NR-1]=$4 " " $5 " " $6
	vel[NR-1]=$7 " " $8 " " $9
	mass[NR-1]=$10
	density[NR-1]=$11
	pressure[NR-1]=$12
	numParts=NR
}

END {
	print "# vtk DataFile Version 2.0"
	print "Converted from TextWriter format"
	print "ASCII\nDATASET UNSTRUCTURED_GRID"

	printf "POINTS %u float\n", NR
	for (i=0; i < numParts; ++i)
		print pos[i]
	print "\n"

	printf "CELLS %u %u\n", numParts, 2*numParts
	for (i=0; i < numParts; ++i)
		printf "1 %u\n", i
	print "\n"

	printf "CELL_TYPES %u\n", numParts
	for (i=0; i < numParts; ++i)
		print "1"
	print "\n"

	printf "POINT_DATA %u\n", numParts

	print "VECTORS Velocity float"
	for (i=0; i < numParts; ++i)
		print vel[i]
	print "\n"

	print "SCALARS Pressure float"
	print "LOOKUP_TABLE default"
	for (i=0; i < numParts; ++i)
		print pressure[i]
	print "\n"

	print "SCALARS Density float"
	print "LOOKUP_TABLE default"
	for (i=0; i < numParts; ++i)
		print density[i]
	print "\n"

	print "SCALARS Mass float"
	print "LOOKUP_TABLE default"
	for (i=0; i < numParts; ++i)
		print mass[i]
	print "\n"

	print "SCALARS Type int"
	print "LOOKUP_TABLE default"
	for (i=0; i < numParts; ++i)
		print type[i]
	print "\n"

	print "SCALARS Object int"
	print "LOOKUP_TABLE default"
	for (i=0; i < numParts; ++i)
		print object[i]
	print "\n"

	print "SCALARS ParticleId int"
	print "LOOKUP_TABLE default"
	for (i=0; i < numParts; ++i)
		print id[i]
	print "\n"
}
