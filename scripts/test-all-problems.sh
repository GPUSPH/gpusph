#!/bin/sh

# Run all problems in GPUSPH, for the given number of iterations (default 10,000), creating simulations with the specified suffix
# Syntax: scripts/test-all-problems.sh suffix [maxiter]

abort() {
	echo "$@" >&2
	exit 1
}

sfx="$1"
maxiter="$2"

# A suffix must be specified
[ -z "$sfx" ] && abort "please specify a suffix"

[ -z "$maxiter" ] && maxiter=10000

for problem in $(make list-problems) ; do
	echo "Running test ${problem} ..."
	make $problem && ./GPUSPH --dir tests/${problem}_${sfx} --maxiter $maxiter || abort "Failed!"
done
