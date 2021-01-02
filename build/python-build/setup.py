#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
from sys import argv
isST=False;
if "--setuptools" in argv:
	argv.remove("--setuptools")
	isST=True;
	from setuptools import setup, Extension
else:
	from distutils.core import setup, Extension

import platform
systemName=platform.system();
import sys
is_64bits = sys.maxsize > 2**32;

with open("README.md", "r") as fh:
	long_description = fh.read()

versionExtention='';
if "Test" in os.environ.get('GITHUB_WORKFLOW',''):
	versionExtention='.dev'+os.environ.get('GITHUB_RUN_NUMBER','');

withEmbedded_JSONLib=True 
if os.path.isfile("../../jsoncpp-master/dist/jsoncpp.cpp"):
	extraCPP=["../../jsoncpp-master/dist/jsoncpp.cpp"];
	extraLib=[];
else:
	extraCPP=[];
	extraLib=['jsoncpp'];

if(systemName=='Darwin' or systemName=='Linux'):
	import numpy.distutils.misc_util
	extra='';
	# if(systemName=='Linux' and platform.python_version()<'3.7'):
	# 	extra='\\';
	setup(name='G2S',
		version=open('../../version', 'r').read()+versionExtention,
		description='G2S interface',
		long_description=long_description,
		**{'long_description_content_type':'text/markdown'} if isST else {},
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
		ext_package = 'g2s',
		ext_modules=[Extension("g2s", sources=["../../src_interfaces/python3_interface.cpp","../../src/DataImage.cpp"]+extraCPP,
			language="c++", 
			extra_compile_args=["-std=c++17",'-DVERSION='+extra+'\"'+open('../../version', 'r').read()+extra+'\"','-DPYTHON_VERSION='+extra+'\"'+platform.python_version()+extra+'\"'],
			extra_link_args=["-std=c++17"],
			include_dirs=["../../include","../../include_interfaces", "/usr/include","/usr/include/jsoncpp","/opt/local/include","../../jsoncpp-master/dist/"],
			libraries = ['zmq']+extraLib,
			library_dirs = ['/usr/lib','/opt/local/lib']
			)],
		include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
	)

if(systemName=='Windows'):
	import numpy
	if is_64bits:
		setup(name='G2S',
			version=open('../../version', 'r').read()+versionExtention,
			description='G2S interface',
			long_description=long_description,
			**{'long_description_content_type':'text/markdown'} if isST else {},
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
			ext_package = 'g2s',
			ext_modules=[Extension("g2s", sources=["../../src_interfaces/python3_interface.cpp","jsoncpp-master/dist/jsoncpp.cpp"],
				language="c++", 
				extra_compile_args=["/std:c++17","-DNOMINMAX",'/DVERSION=\\\"'+open('../../version', 'r').read()+'\\\"','/DPYTHON_VERSION=\\\"'+platform.python_version()+'\\\"'],
				extra_link_args=[],
				include_dirs=["../../include","./cppzmq-master","jsoncpp-master/include", "../../include_interfaces", "./libzmq-v141-x64-4_3_2", "jsoncpp-master\\jsoncpp-master\\include", "/usr/include","/usr/include/jsoncpp","/opt/local/include"],
				libraries = ['libzmq-v141-mt-4_3_2'],
				library_dirs = ['/usr/lib','/opt/local/lib',"./libzmq-v141-x64-4_3_2"]
			)],
			include_dirs=numpy.get_include(),
			data_files=[('lib\\site-packages\\g2s',["./libzmq-v141-x64-4_3_2/libzmq-v141-mt-4_3_2.dll","./libzmq-v141-x64-4_3_2/libsodium.dll"])]
		);
	else:
		setup(name='G2S',
			version=open('../../version', 'r').read()+versionExtention,
			description='G2S interface',
			long_description=long_description,
			**{'long_description_content_type':'text/markdown'} if isST else {},
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
			ext_package = 'g2s',
			ext_modules=[Extension("g2s", sources=["../../src_interfaces/python3_interface.cpp","jsoncpp-master/dist/jsoncpp.cpp"],
				language="c++", 
				extra_compile_args=["/std:c++17","-DNOMINMAX",'/DVERSION=\\\"'+open('../../version', 'r').read()+'\\\"','/DPYTHON_VERSION=\\\"'+platform.python_version()+'\\\"'],
				extra_link_args=[],
				include_dirs=["../../include","./cppzmq-master","jsoncpp-master/include", "../../include_interfaces", "./libzmq-v141-4_3_2", "jsoncpp-master\\jsoncpp-master\\include", "/usr/include","/usr/include/jsoncpp","/opt/local/include"],
				libraries = ['libzmq-v141-mt-4_3_2'],
				library_dirs = ['/usr/lib','/opt/local/lib',"./libzmq-v141-4_3_2"]
			)],
			include_dirs=numpy.get_include(),
			data_files=[('lib\\site-packages\\g2s',["./libzmq-v141-4_3_2/libzmq-v141-mt-4_3_2.dll","./libzmq-v141-4_3_2/libsodium.dll"])]
		);
