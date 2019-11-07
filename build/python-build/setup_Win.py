# -*- coding: UTF-8 -*-

#py setup_Win.py bdist --format=wininst

from distutils.core import setup, Extension
import numpy

setup(name='G2S',
      version='0.94.0',
      description='G2S interface',
      author='Mathieu Gravey',
      author_email='mathieu.gravey@unil.ch',
      url='https://github.com/GAIA-UNIL/G2S',
      ext_modules=[Extension("g2s", sources=["../../src_interfaces/python3_interface.cpp","jsoncpp-master/dist/jsoncpp.cpp"],
        language="c++", 
        extra_compile_args=["/std:c++17","-DNOMINMAX"],
        extra_link_args=[],
        include_dirs=["../../include","./cppzmq-master","jsoncpp-master/include", "../../include_interfaces", "C:\Program Files\ZeroMQ 4.0.4\include", "jsoncpp-master\\jsoncpp-master\\include", "/usr/include","/usr/include/jsoncpp","/opt/local/include"],
        libraries = ['libzmq-v120-mt-4_0_4'],
        library_dirs = ['/usr/lib','/opt/local/lib',"C:\Program Files\ZeroMQ 4.0.4\lib"]
      )],
    include_dirs=numpy.get_include(),
    data_files=[('lib\\site-packages\\',["C:\\Program Files\\ZeroMQ 4.0.4\\bin\\libzmq-v120-mt-4_0_4.dll","C:\\Windows\\System32\\msvcr120.dll","C:\\Windows\\System32\\msvcp120.dll"])]
);

# py -3.8 -m pip install --upgrade pip
# py -3.7 -m pip install --upgrade pip
# py -3.6 -m pip install --upgrade pip
# py -3.5 -m pip install --upgrade pip

# py -3.8 -m pip install numpy setuptools wheel
# py -3.7 -m pip install numpy setuptools wheel
# py -3.6 -m pip install numpy setuptools wheel
# py -3.5 -m pip install numpy setuptools wheel

# py -3.8 setup_Win.py bdist --format=wininst
# py -3.7 setup_Win.py bdist --format=wininst
# py -3.6 setup_Win.py bdist --format=wininst
# py -3.5 setup_Win.py bdist --format=wininst