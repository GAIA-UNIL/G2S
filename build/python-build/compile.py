#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy.distutils.misc_util

print(numpy.distutils.misc_util.get_numpy_include_dirs())

setup(
    ext_modules=[Extension("g2s", sources=["../../src/g2smodule.cpp"],
    	language="c++", 
    	extra_compile_args=["-std=c++11"],
    	extra_link_args=["-std=c++11"],
    	include_dirs=["../../include", "/usr/include","/usr/include/jsoncpp","/opt/local/include"],
    	libraries = ['zmq','jsoncpp'],
        library_dirs = ['/usr/lib','/opt/local/lib']
         )],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)