# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy.distutils.misc_util

print(numpy.distutils.misc_util.get_numpy_include_dirs())

setup(name='G2S',
      version='0.9.0',
      description='G2S interface',
      author='Mathieu Gravey',
      author_email='mathieu.gravey@unil.ch',
      url='https://github.com/GAIA-UNIL/G2S',
    ext_modules=[Extension("g2s", sources=["../../src/g2smodule.cpp","jsoncpp-master\\jsoncpp-master/dist/jsoncpp.cpp"],
    	language="c++", 
    	extra_compile_args=["-std=c++11"],
    	extra_link_args=["-std=c++11",'-static'],
    	include_dirs=["../../include",'.',"C:\Program Files\ZeroMQ 4.0.4\include", "jsoncpp-master\\jsoncpp-master\\include", "/usr/include","/usr/include/jsoncpp","/opt/local/include"],
    	libraries = ['libzmq-v120-mt-4_0_4'],
        library_dirs = ['/usr/lib','/opt/local/lib',"C:\Program Files\ZeroMQ 4.0.4\lib"]
         )],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
    data_files=[('lib\\site-packages\\',["C:\\Program Files\\ZeroMQ 4.0.4\\bin\\libzmq-v120-mt-4_0_4.dll"])]