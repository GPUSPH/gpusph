#!/bin/sh

# Run all problems in GPUSPH, for the given number of iterations, creating simulations with the specified suffix
# Syntax: run-all-problems [suffix] [maxiter]
# default suffix is 'reference'
# default maxiter is 1000

abort() {
	echo "$@" >&2
	exit 1
}

sfx="$1"
maxiter="$2"

[ -z "$sfx" ] && sfx=reference
[ -z "$maxiter" ] && maxiter=1000

for problem in $(make list-problems) ; do
	echo "Running test ${problem} ..."
	outdir="tests/${problem}_${sfx}"
	if [ -d "$outdir" ] ; then
		echo "$outdir exists, skipping"
		continue
	fi
	make $problem && ./GPUSPH --dir "$outdir" --maxiter $maxiter || abort "Failed!"
done
