clc; clear all;

% Builds the g2s MEX file.
%   Default: C++ MEX API interface (src_interfaces/matlab_interface_modern.cpp, R2018a+).
%   Set the environment variable G2S_MATLAB_LEGACY_API=1 to build the C API interface instead.
% jsoncpp (amalgamated) and cppzmq are vendored in include_interfaces/third_party and compiled
% into the MEX file, so the result does not depend on a jsoncpp shared library.

useLegacyApi=strcmp(getenv('G2S_MATLAB_LEGACY_API'),'1');
if useLegacyApi
    interfaceSource='../../src_interfaces/matlab_interface.cpp';
    % Recent mex releases add the C++ MEX entry points to the export list of every C++ source,
    % which the current Apple linker rejects for C API MEX files: drop that export list.
    apiOptions={'LINKEXPORTCPP='};
else
    interfaceSource='../../src_interfaces/matlab_interface_modern.cpp';
    apiOptions={'-R2018a'};
end

versionString=strtrim(fileread('../../version'));
thirdParty='../../include_interfaces/third_party';
commonSources={interfaceSource,'../../src_interfaces/browserTransport.cpp',[thirdParty '/jsoncpp/jsoncpp.cpp']};
% Vendored headers first so they win over any system jsoncpp/cppzmq.
commonIncludes={['-I' thirdParty '/jsoncpp'],['-I' thirdParty '/cppzmq'],'-I../../include_interfaces','-I../../include'};
commonDefines={'-DG2S_ENABLE_BROWSER_TRANSPORT=1',strcat('-DMATLAB_VERSION=0x',version('-release')),['-DVERSION=\"' versionString '\"']};

if ismac
    % Apple Silicon with Homebrew (tested). Intel Macs are not supported explicitly.
    % libzmq and libsodium are linked statically from Homebrew.
    prefix='/opt/homebrew';
    macMin='13.0';
    mex('-output','g2s',commonSources{:},commonIncludes{:},['-I' prefix '/include'],commonDefines{:},...
        [prefix '/lib/libzmq.a'],[prefix '/lib/libsodium.a'],'-lut',apiOptions{:},...
        ['CXXFLAGS=$CXXFLAGS -std=c++17 -mmacosx-version-min=' macMin],['LDFLAGS=$LDFLAGS -mmacosx-version-min=' macMin]);
elseif isunix
    zmqLib='/usr/lib/x86_64-linux-gnu/libzmq.a';
    if ~isfile(zmqLib), zmqLib='/usr/local/lib/libzmq.a'; end
    sodiumLib='/usr/lib/x86_64-linux-gnu/libsodium.a';
    if ~isfile(sodiumLib), sodiumLib='/usr/local/lib/libsodium.a'; end
    mex('-output','g2s',commonSources{:},commonIncludes{:},'-I/usr/local/include','-I/usr/include',commonDefines{:},...
        zmqLib,sodiumLib,'-lut',apiOptions{:},'CXXFLAGS=$CXXFLAGS -std=c++17','LDFLAGS=$LDFLAGS');
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
    libsVal=cellstr(ls(strcat(zmqBuilDir,'lib/Release')));
    libName=libsVal(contains(libsVal,'.lib')&contains(libsVal,'libzmq')&(~contains(libsVal,'mt-s')));
    mex('-output','g2s',commonSources{:},commonIncludes{:},'-I./libzmq-master/include',commonDefines{:},...
        strcat('-L',zmqBuilDir,'lib/Release'),strcat('-l',libName{1}(4:end-4)),strcat('-L',matlabroot,'\extern\lib\win64\microsoft'),...
        '-lut','-lws2_32','-DNOMINMAX','-D_USE_MATH_DEFINES',apiOptions{:},'COMPFLAGS=$COMPFLAGS /std:c++17');
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
