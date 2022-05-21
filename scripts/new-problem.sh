#!/bin/sh

abort() {
	printf "%s\n" "$1" >&2
	exit 1
}

must_exist() {
	fname="$1"
	[ -e "$fname" ] || abort "Source file '${fname}' does not exist"
}

must_not_exist() {
	fname="$1"
	[ -e "$fname" ] && abort "Destination file '${fname}' exists already"
}


problem_name="$1"

[ -z "$problem_name" ] && abort "Please specfiy a problem name"

# Requirement: problem name must begin with a letter and can only contain letters, numbers and _ sign

clean="$(printf "%s" "$problem_name" | grep -o -E '^[A-Za-z][A-Za-z0-9_]+$')"

[ "x${clean}" = "x${problem_name}" ] || abort "Problem name must begin with a letter and can only consist of letters, numbers and the underscore sign _"

problem_name_ucase="$(printf "%s" "$problem_name" | tr '[a-z]' '[A-Z]')"

src_h="src/problems/ProblemTemplate.h.template"
src_cc="src/problems/ProblemTemplate.cc.template"
src_cu="src/problems/ProblemTemplate.cu.template"

must_exist "${src_h}"
must_exist "${src_cc}"
must_exist "${src_cu}"

dst_h="src/problems/${problem_name}.h"
dst_cc="src/problems/${problem_name}.cc"
dst_cu="src/problems/${problem_name}.cu"

must_not_exist "${dst_h}"
must_not_exist "${dst_cc}"
must_not_exist "${dst_cu}"

sed -e "s/ProblemTemplate/${problem_name}/g;s/PROBLEMTEMPLATE/${problem_name_ucase}/g" "${src_h}"  > "${dst_h}"
sed -e "s/ProblemTemplate/${problem_name}/g;s/PROBLEMTEMPLATE/${problem_name_ucase}/g" "${src_cc}" > "${dst_cc}"
sed -e "s/ProblemTemplate/${problem_name}/g;s/PROBLEMTEMPLATE/${problem_name_ucase}/g" "${src_cu}" > "${dst_cu}"
