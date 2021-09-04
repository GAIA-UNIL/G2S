if NOT exist "libzmq" (
	git clone https://github.com/zeromq/libzmq
	cd libzmq
	mkdir action_build
	cd action_build
	cmake ..
	msbuild ZeroMQ.sln /property:Configuration=Release /m:4
)

if NOT exist "cppzmq-master" (
	powershell -Command "(New-Object Net.WebClient).DownloadFile('https://codeload.github.com/zeromq/cppzmq/zip/master', 'cppzmq-master.zip')"
	powershell Expand-Archive cppzmq-master.zip -DestinationPath .
)

if NOT exist "jsoncpp-master" (
	powershell -Command "(New-Object Net.WebClient).DownloadFile('https://codeload.github.com/open-source-parsers/jsoncpp/zip/master', 'jsoncpp-master.zip')"
	powershell Expand-Archive jsoncpp-master.zip -DestinationPath .
	cd jsoncpp-master
	py amalgamate.py
	cd ..
)