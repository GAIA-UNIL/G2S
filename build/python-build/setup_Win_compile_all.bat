@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem ----- Build the shared libzmq dependency used by the Windows wheel -----
if NOT exist "libzmq\CMakeLists.txt" (
	git clone --depth 1 --branch v4.3.5 https://github.com/zeromq/libzmq.git libzmq
	if errorlevel 1 exit /b 1
)

cmake -S libzmq -B libzmq\build -A x64 ^
	-DCMAKE_BUILD_TYPE=Release ^
	-DBUILD_SHARED=ON ^
	-DBUILD_STATIC=OFF ^
	-DBUILD_TESTS=OFF ^
	-DWITH_DOCS=OFF ^
	-DWITH_PERF_TOOL=OFF
if errorlevel 1 exit /b 1

cmake --build libzmq\build --config Release --parallel
if errorlevel 1 exit /b 1

if NOT exist "libzmq\build\lib\Release\libzmq*.lib" (
	echo ERROR: libzmq import library was not produced.
	exit /b 1
)
if NOT exist "libzmq\build\bin\Release\libzmq*.dll" (
	echo ERROR: libzmq runtime DLL was not produced.
	exit /b 1
)

rem PEP 517 may build in a temporary directory, so publish the absolute
rem dependency root to subsequent GitHub Actions steps.
if defined GITHUB_ENV (
	echo G2S_LIBZMQ_ROOT=%CD%\libzmq>>"%GITHUB_ENV%"
)

echo libzmq built and ready at %CD%\libzmq
endlocal
