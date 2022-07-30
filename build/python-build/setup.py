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

if os.path.isfile("../../jsoncpp-master/dist/jsoncpp.cpp"):
	extraCPP=["../../jsoncpp-master/dist/jsoncpp.cpp"];
	extraLib=['z'];
else:
	extraCPP=[];
	extraLib=['jsoncpp','z'];

extra='';

if(systemName=='Darwin' or systemName=='Linux'):
	import numpy.distutils.misc_util
	if(platform.python_version()<'3.8'):
		extra='\\';
	setup(name='G2S',
		version=open('../../version', 'r').read()+versionExtention,
		description='G2S interface',
		long_description=long_description,
		**{'long_description_content_type':'text/markdown'} if isST else {},
		author='Mathieu Gravey',
		author_email='g2s@mgravey.com',
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
			include_dirs=[numpy.get_include(),"../../include","../../include_interfaces", "/usr/include","/usr/include/jsoncpp","/opt/local/include","../../jsoncpp-master/dist/"],
			libraries = ['zmq']+extraLib,
			library_dirs = ['/usr/lib','/opt/local/lib']
			)],
		include_dirs=numpy.get_include(),
		install_requires=['numpy']
	)

if(systemName=='Windows'):
	from packaging import version
	if version.parse(platform.python_version())<version.parse('3.9'):
		extra='\\';
	import numpy
	zmqBuilDir="./libzmq/action_build/";
	if is_64bits:
		setup(name='G2S',
			version=open('../../version', 'r').read()+versionExtention,
			description='G2S interface',
			long_description=long_description,
			**{'long_description_content_type':'text/markdown'} if isST else {},
			author='Mathieu Gravey',
			author_email='g2s@mgravey.com',
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
				extra_compile_args=["/std:c++17","-DNOMINMAX",'/DVERSION='+extra+'\"'+open('../../version', 'r').read()+extra+'\"','/DPYTHON_VERSION='+extra+'\"'+platform.python_version()+extra+'\"'],
				extra_link_args=[],
				include_dirs=["../../include","./cppzmq-master","jsoncpp-master/include", "../../include_interfaces", "libzmq/include", "jsoncpp-master\\jsoncpp-master\\include", "/usr/include","/usr/include/jsoncpp","/opt/local/include"],
				libraries = [x[:-4] for x in os.listdir(zmqBuilDir+"lib/Release") if 'mt-s' not in x and '.lib' in x and 'libzmq' in x ],
				library_dirs = ['/usr/lib','/opt/local/lib',zmqBuilDir+"lib/Release"]
			)],
			include_dirs=numpy.get_include(),
			install_requires=['numpy'],
			data_files=[('lib\\site-packages\\g2s', [zmqBuilDir+"bin/Release/"+x for x in os.listdir(zmqBuilDir+"bin/Release") if 'mt-s' not in x and '.dll' in x and 'libzmq' in x ])]
		);
	else:
		setup(name='G2S',
			version=open('../../version', 'r').read()+versionExtention,
			description='G2S interface',
			long_description=long_description,
			**{'long_description_content_type':'text/markdown'} if isST else {},
			author='Mathieu Gravey',
			author_email='g2s@mgravey.com',
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
				extra_compile_args=["/std:c++17","-DNOMINMAX",'/DVERSION='+extra+'\"'+open('../../version', 'r').read()+extra+'\"','/DPYTHON_VERSION='+extra+'\"'+platform.python_version()+extra+'\"'],
				extra_link_args=[],
				include_dirs=["../../include","./cppzmq-master","jsoncpp-master/include", "../../include_interfaces", "libzmq/include", "jsoncpp-master\\jsoncpp-master\\include", "/usr/include","/usr/include/jsoncpp","/opt/local/include"],
				libraries = [x[:-4] for x in os.listdir(zmqBuilDir+"lib/Release") if 'mt-s' not in x and '.lib' in x and 'libzmq' in x ],
				library_dirs = ['/usr/lib','/opt/local/lib',zmqBuilDir+"lib/Release"]
			)],
			include_dirs=numpy.get_include(),
			install_requires=['numpy'],
			data_files=[('lib\\site-packages\\g2s', [zmqBuilDir+"bin/Release/"+x for x in os.listdir(zmqBuilDir+"bin/Release") if 'mt-s' not in x and '.dll' in x and 'libzmq' in x ])]
		);
