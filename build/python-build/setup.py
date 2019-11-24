#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from sys import argv
if "--setuptools" in argv:
	argv.remove("--setuptools")
	from setuptools import setup, Extension
else:
	from distutils.core import setup, Extension

import platform
systemName=platform.system();

with open("README.md", "r") as fh:
	long_description = fh.read()

if(systemName=='Darwin' or systemName=='Linux'):
	import numpy.distutils.misc_util
	extra='';
	if(systemName=='Linux' and platform.python_version()<'3.7'):
		extra='\\';
	setup(name='G2S',
		version=open('../../version', 'r').read()+'.dev10',
		description='G2S interface',
		long_description=long_description,
		author='Mathieu Gravey',
		author_email='mathieu.gravey@unil.ch',
		url='https://github.com/GAIA-UNIL/G2S',
		license='GPLv3',
		packages=['g2s'],
		classifiers=[
			'Development Status :: 3 - Alpha',
			'Intended Audience :: Science/Research',
			'Intended Audience :: Education',
			'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
			'Programming Language :: C++',
			'Programming Language :: Python :: 3 :: Only'
		],
		ext_modules=[Extension("g2s/g2s", sources=["../../src_interfaces/python3_interface.cpp"],
			language="c++", 
			extra_compile_args=["-std=c++17",'-DVERSION='+extra+'\"'+open('../../version', 'r').read()+extra+'\"','-DPYTHON_VERSION=\"'+platform.python_version()+'\"'],
			extra_link_args=["-std=c++17"],
			include_dirs=["../../include","../../include_interfaces", "/usr/include","/usr/include/jsoncpp","/opt/local/include"],
			libraries = ['zmq','jsoncpp'],
			library_dirs = ['/usr/lib','/opt/local/lib']
			)],
		include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
	)

if(systemName=='Windows'):
	import numpy
	setup(name='G2S',
		version=open('../../version', 'r').read()+'.dev10',
		description='G2S interface',
		long_description=long_description,
		author='Mathieu Gravey',
		author_email='mathieu.gravey@unil.ch',
		url='https://github.com/GAIA-UNIL/G2S',
		license='GPLv3',
		packages=['g2s'],
		classifiers=[
			'Development Status :: 3 - Alpha',
			'Intended Audience :: Science/Research',
			'Intended Audience :: Education',
			'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
			'Programming Language :: C++',
			'Programming Language :: Python :: 3 :: Only'
		],
		ext_modules=[Extension("g2s", sources=["../../src_interfaces/python3_interface.cpp","jsoncpp-master/dist/jsoncpp.cpp"],
			language="c++", 
			extra_compile_args=["/std:c++17","-DNOMINMAX",'/DVERSION=\\\"'+open('../../version', 'r').read()+'\\\"','/DPYTHON_VERSION=\\\"'+platform.python_version()+'\\\"'],
			extra_link_args=[],
			include_dirs=["../../include","./cppzmq-master","jsoncpp-master/include", "../../include_interfaces", "C:\Program Files\ZeroMQ 4.0.4\include", "jsoncpp-master\\jsoncpp-master\\include", "/usr/include","/usr/include/jsoncpp","/opt/local/include"],
			libraries = ['libzmq-v120-mt-4_0_4'],
			library_dirs = ['/usr/lib','/opt/local/lib',"C:\Program Files\ZeroMQ 4.0.4\lib"]
		)],
		include_dirs=numpy.get_include(),
		data_files=[('lib\\site-packages\\',["C:\\Program Files\\ZeroMQ 4.0.4\\bin\\libzmq-v120-mt-4_0_4.dll","C:\\Windows\\System32\\msvcr120.dll","C:\\Windows\\System32\\msvcp120.dll"])]
	);
