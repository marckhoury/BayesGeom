#!/usr/bin/env python

import os, sys
import commands

which_python = 'python' + sys.version[:3].strip()

python_prefix = commands.getoutput('python-config --prefix').strip()

if os.path.exists('{0}/Python'.format(python_prefix)):
    args = "-DPYTHON_LIBRARY='{0}/Python'".format(python_prefix)
    args += " -DPYTHON_INCLUDE_DIR='{0}/Headers'".format(python_prefix)
else:
    python_lib = "{0}/lib/lib{1}".format(python_prefix, which_python)
    if os.path.exists('{0}.a'.format(python_lib)):
        args = "-DPYTHON_LIBRARY='{0}.a'".format(python_lib)
    else:
        args = "-DPYTHON_LIBRARY='{0}.dylib'".format(python_lib)
    args += " -DPYTHON_PACKAGES_PATH='{0}/{1}/site-packages'".format(python_lib, which_python)
    
os.system('cmake . {0}'.format(args))
os.system('make')