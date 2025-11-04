clc; clear all;

communMacLinux={'-output','g2s','../../src_interfaces/matlab_interface.cpp','-I../../include_interfaces','-I../../include','-I/usr/local/include','-I/usr/include/jsoncpp','-I/usr/include','-I/opt/local/include','-I/opt/homebrew/include',...
    '/usr/local/lib/libzmq.a','/opt/local/lib/libjsoncpp.a','/usr/local/lib/libsodium.a','-lut',strcat('-DMATLAB_VERSION=0x',version('-release')),strcat('-DVERSION=\"',fileread('../../version'),'\"')};

if ismac
    if computer('arch')=="maci64"
        mex(communMacLinux{:},'-L/opt/local/lib/','-L/usr/local/lib/','CXXFLAGS=$CXXFLAGS -std=c++17 -mmacosx-version-min=10.14', 'LDFLAGS=$LDFLAGS -mmacosx-version-min=10.14');
    end
    if computer('arch')=="maca64"
        communMacLinux={'-output','g2s','../../src_interfaces/matlab_interface_modern.cpp','-I../../include_interfaces','-I../../include','-I/usr/local/include','-I/usr/include/jsoncpp','-I/usr/include','-I/opt/local/include','-I/opt/homebrew/include',...
    '/opt/homebrew/lib/libzmq.a','-ljsoncpp','/opt/homebrew/lib/libsodium.a','-lut',strcat('-DMATLAB_VERSION=0x',version('-release')),strcat('-DVERSION=\"',fileread('../../version'),'\"')};
        mex(communMacLinux{:},"-L/opt/homebrew/lib/",'CXXFLAGS=$CXXFLAGS -std=c++17 -mmacosx-version-min=15.0', 'LDFLAGS=$LDFLAGS -mmacosx-version-min=15.0','-R2018a');
    end
elseif isunix
    mex(communMacLinux{:},'-L/usr/lib','CXXFLAGS=$CXXFLAGS -std=c++17 ', 'LDFLAGS=$LDFLAGS');
elseif ispc
    zmqBuilDir='./libzmq-master/build4Matlab/';
    if(exist('libzmq-master')==0)
        websave('libzmq.zip','https://github.com/zeromq/libzmq/archive/refs/heads/master.zip');
        unzip('libzmq.zip');
        cd libzmq-master
        mkdir build4Matlab
        cd build4Matlab
        !cmake ..
        setenv('PATH',strcat(getenv('PATH'),'c:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin',';'))
        !msbuild ZeroMQ.sln /property:Configuration=Release /m:4
        cd ../..
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
    libsVal=cellstr(ls(strcat(zmqBuilDir,'lib/Release')));
    libName=libsVal(contains(libsVal,'.lib')&contains(libsVal,'libzmq')&(~contains(libsVal,'mt-s')));
    mex('-output','g2s','../../src_interfaces/matlab_interface.cpp','jsoncpp-master/dist/jsoncpp.cpp','-I../../include_interfaces','-I../../include','-I./libzmq-master/include',...
        '-I"cppzmq-master"',strcat('-L',zmqBuilDir,'lib/Release'),strcat('-l',libName{1}(4:end-4)),'-I"jsoncpp-master\dist"',strcat('-L',matlabroot,'\extern\lib\win64\microsoft'),...
        '-lut','-DNOMINMAX',strcat('-DVERSION=\"',fileread('../../version'),'\"'),'-D_USE_MATH_DEFINES',strcat('-DMATLAB_VERSION=0x',version('-release')),'COMPFLAGS=$COMPFLAGS /std:c++17');
    path=getenv('PATH');
    newpath=strcat(path,strcat('-I',zmqBuilDir,'bin/Release'));
    setenv('PATH',newpath);
    copyfile(strcat(zmqBuilDir,'bin/Release/',libName{1}(1:end-3),'dll'))
    zip('./../../../G2S-compiled-interfaces/latest/MATLAB/Windows/G2S-latest.win-amd64-matlab.zip',{'g2s.mexw64',strcat(libName{1}(1:end-3),'dll')});
    mkdir('./../../../G2S-compiled-interfaces/',strcat(strjoin(string(fscanf(fopen('../../version'),'%d.%d\n')),'.'),'/MATLAB/Windows/'));
    copyfile('./../../../G2S-compiled-interfaces/latest/MATLAB/Windows/G2S-latest.win-amd64-matlab.zip',strcat('./../../../G2S-compiled-interfaces/',strjoin(string(fscanf(fopen('../../version'),'%d.%d\n')),'.'),'/MATLAB/Windows/','G2S-',fscanf(fopen('../../version'),'%s\n'),'.win-amd64-matlab.zip'))
    
else
    disp('Platform not supported')
end