#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy.distutils.misc_util
import platform

print(numpy.distutils.misc_util.get_numpy_include_dirs())

setup(name='G2S',
	version=open('../../version', 'r').read(),
	description='G2S interface',
	author='Mathieu Gravey',
	author_email='mathieu.gravey@unil.ch',
	url='https://github.com/GAIA-UNIL/G2S',
	ext_modules=[Extension("g2s", sources=["../../src_interfaces/python3_interface.cpp"],
		language="c++", 
		extra_compile_args=["-std=c++17",'-DVERSION=\"'+open('../../version', 'r').read()+'\"','-DPYTHON_VERSION=\"'+platform.python_version()+'\"'],
		extra_link_args=["-std=c++17"],
		include_dirs=["../../include","../../include_interfaces", "/usr/include","/usr/include/jsoncpp","/opt/local/include"],
		libraries = ['zmq','jsoncpp'],
		library_dirs = ['/usr/lib','/opt/local/lib']
		)],
	include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)