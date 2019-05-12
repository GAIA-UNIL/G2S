clc; clear all;

communMacLinux={'-output','g2s','../../src_interfaces/matlab_interface.cpp','-I../../include_interfaces','-I../../include','-I/usr/local/include','-I/usr/include','-I/opt/local/include','-lzmq','-ljsoncpp','-lut',strcat('-DMATLAB_VERSION=0x',version('-release'))};

if ismac
    mex(communMacLinux{:},'-L/opt/local/lib/','CXXFLAGS=$CXXFLAGS -std=c++17 -mmacosx-version-min=10.14', 'LDFLAGS=$LDFLAGS -mmacosx-version-min=10.14');
elseif isunix
    mex(communMacLinux{:},'-L/usr/lib','CXXFLAGS=$CXXFLAGS -std=c++17 ', 'LDFLAGS=$LDFLAGS');
elseif ispc
    if(exist('C:\Program Files\ZeroMQ 4.0.4')==0)
        websave('ZeroMQ-4.0.4~miru1.0-x64.exe','https://miru.hk/archive/ZeroMQ-4.0.4~miru1.0-x64.exe');
        winopen('ZeroMQ-4.0.4~miru1.0-x64.exe')
    end
    if(exist('cppzmq-master')==0)
        websave('cppzmq-master.zip','https://codeload.github.com/zeromq/cppzmq/zip/master');
        unzip('cppzmq-master.zip');
    end
    if(exist('jsoncpp-master')==0)
        websave('jsoncpp-master.zip','https://codeload.github.com/open-source-parsers/jsoncpp/zip/master')
        unzip('jsoncpp-master.zip');
        cd 'jsoncpp-master'
        ! python amalgamate.py
        cd ..
    end
    mex('-output','g2s','../../src_interfaces/matlab_interface.cpp','jsoncpp-master/dist/jsoncpp.cpp','-I../../include_interfaces','-I../../include','-I"C:\Program Files\ZeroMQ 4.0.4\include"','-I"cppzmq-master"','-L"C:\Program Files\ZeroMQ 4.0.4\lib"','-llibzmq-v120-mt-4_0_4','-I"jsoncpp-master\dist"',strcat("-L",matlabroot,"\extern\lib\win64\microsoft"),'-lut','-DNOMINMAX','-D_USE_MATH_DEFINES',strcat('-DMATLAB_VERSION=0x',version('-release')),'COMPFLAGS=$COMPFLAGS /std:c++17');
    path=getenv('PATH');
    newpath=strcat(path,'C:\Program Files\ZeroMQ 4.0.4\bin;');
    setenv('PATH',newpath);
else
    disp('Platform not supported')
end