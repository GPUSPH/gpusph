#!/bin/sh

# This script parses the problem files looking for the engine instances needed.

# We do some in-place editing with sed, that as a slightly different syntax
# in Darwin (-i requires a suffix)

if [ "$(uname -s 2> /dev/null)" = Darwin ] ; then
	sed_i="sed -i ''"
else
	sed_i="sed -i"
fi

# Get the defaults first.
# Sets variables in the form default_{typename}, e.g. default_KernelType or
# default_flag_t
get_defaults() {
	eval $(sed -n -e '/struct TypeDefaults/,/};/ p' src/cuda/cudasimframework.cu | grep typedef | sed -e 's/	typedef TypeValue</default_/' -e 's/, /="/' -e 's/>.*/"/')
}

# reset kernel, formulation etc to default
reset_to_default() {
	kernel="${default_KernelType}"
	formulation="${default_SPHFormulation}"
	viscosity="${default_ViscosityType}"
	boundary="${default_BoundaryType}"
	periodicity="${default_Periodicity}"
	flags="${default_flag_t}"
}

# add a single instance of a given engine in a given context to a given file
# reads globals $instance $file and $context
add_instance() {
	# DEBUG
	# echo "adding '$instance' to '$file' because of '$context'"

	test -z "$file" && { echo "No file !!!" ; exit 1 ; }

	# make sure directory exists
	mkdir -p "$(dirname "$file")"

	# create file if missing
	test -e "$file" || touch "$file"

	# check if instance is there, and get the line number
	line="$(grep -n "$instance" "$file" | cut -f1 -d:)"

	if [ -z "$line" ] ; then
		# not found, add instance and comment
		echo "// $context" >> "$file"
		echo "$instance" >> "$file"
	else
		# instance is there already, was it added for this same context?
		found="$(sed -n -e "$(($line - 1))p" "$file" | grep " $context" || true)"
		# if the context is not found in the previous line (the comment)
		# add it
		if test -z "$found" ; then
			$sed_i -e "$(($line - 1)) s/$/, $context/" "$file"
		fi
	fi

}

# add the specified instances for each engine
add_instances() {
	context="$1"

	# sort flags so that they are always in the same sequence:
	# split flags into one per line, sort, join lines together again.
	# Since sed in Mac OS X doesn't accept \n for newline, we insert a verbatim
	# newline. Also, in Mac OS X it seems to be impossible to replace newlines
	# with something else, so we use awk to join the lines and then sed
	# to remove the last end-of-record echoed by awk
	sortflags=$(echo ${flags} | sed 's/[ 	]*|[ 	]*/\
/g' | sort | awk 1 ORS=' | ' | sed 's/ | $//')

	# neibs engine
	file="$BUILDNEIBS_INSTANCE_FILE"
	instance="template class CUDANeibsEngine<${boundary}, ${periodicity}, true>;"
	add_instance

	# integration engine
	file="$EULER_INSTANCE_FILE"
	xsphcorr=$(echo ${flags} | grep -q ENABLE_XSPH && echo true || echo false)
	instance="template class CUDAPredCorrEngine<${formulation}, ${boundary}, ${xsphcorr}>;"
	add_instance

	# forces engine
	file="$FORCES_INSTANCE_FILE"
	instance="template class CUDAForcesEngine<${kernel}, ${formulation}, ${viscosity}, ${boundary}, ${sortflags}>;"
	add_instance

	# viscosity engine
	file="$VISC_INSTANCE_FILE"
	instance="template struct CUDAViscEngineHelper<${viscosity}, ${kernel}, ${boundary}>;"
	add_instance

	# filters engine
	file="$FILTERS_INSTANCE_FILE"
	instance="template class CUDAFilterEngine<SHEPARD_FILTER, ${kernel}, ${boundary}>;"
	add_instance
	instance="template class CUDAFilterEngineHelper<SHEPARD_FILTER, ${kernel}, ${boundary}>;"
	add_instance
	instance="template class CUDAFilterEngine<MLS_FILTER, ${kernel}, ${boundary}>;"
	add_instance
	instance="template class CUDAFilterEngineHelper<MLS_FILTER, ${kernel}, ${boundary}>;"
	add_instance

	# boundary conditions engine
	file="$BOUND_INSTANCE_FILE"
	# currently only needed if boundary == SA_BOUNDARY
	if [ $boundary = SA_BOUNDARY ] ; then
		instance="template class CUDABoundaryConditionsEngine<${kernel}, ${viscosity}, ${boundary}, ${sortflags}>;"
		add_instance
	fi
}

# Process a single source file, overriding the defaults as needed
process_file() {
	fname="$1"

	# extract framework setup, remove comments, join everything in a single line,
	# split at semi-colons, replace runs of whitespace with single space,
	# remove spaces after < and before >
	# TODO handle conditionals / ifdefs in the same SETUP_FRAMEWORK
	sed -n -e '/SETUP_FRAMEWORK/,/);/ p' "$fname" | \
		sed 's!//.*$!!' | tr '\n' ' ' | tr ';' '\n' | sed 's/[ 	]\{1,\}/ /g' | \
		sed -e 's/< /</g' -e 's/ >/>/g' | \
	while read line ; do
		# at this point we have one framework setup per line, with normalized whitespace.
		# for each of it, extract the template overrides (grep -o)
		# and replace the syntax with one that does assignments
		overrides="$(echo $line | grep -o '\w\+<[^>]\+>' | sed -e 's/</="/' -e 's/>/"/')"

		# actual compute the overrides
		reset_to_default
		eval $overrides

		# add all the instances needed by the given setup
		add_instances "$(basename "$fname" .cc)"
	done
}

get_defaults

reset_to_default
add_instances '(default)'

for source in "$@" ; do
	process_file "$source"
done

# Touch all instance files, to ensure they all exist. This is mostly needed
# to avoid that the boundary instance file is not generated when no problem
# has SA_BOUNDARY conditions
for file in \
	"$BUILDNEIBS_INSTANCE_FILE" \
	"$EULER_INSTANCE_FILE" \
	"$FORCES_INSTANCE_FILE" \
	"$VISC_INSTANCE_FILE" \
	"$FILTERS_INSTANCE_FILE" \
	"$BOUND_INSTANCE_FILE"
do
	touch "$file"
done
