@echo off
setlocal

rem ----- Build libzmq -----
if NOT exist "libzmq" (
	git clone https://github.com/zeromq/libzmq.git
	cd libzmq
	mkdir build && cd build
	cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED=ON
	msbuild ZeroMQ.sln /p:Configuration=Release /m
	cd ../..
)

echo ✅ libzmq built and headers ready
endlocal
