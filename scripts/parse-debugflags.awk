#!/usr/bin/awk

BEGIN {
	print "/*! \\file"
	print " * AUTOGENERATED FILE, DO NOT EDIT, see src/debugflags.def instead."
	print " * A cascade of ifs to parse the debug flags passed on the command line."
	print " */"
	print ""
}

/^unsigned / {
	print "if (flag == \"" $2 "\") ret." $2 " = 1; else "
}
