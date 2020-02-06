if NOT exist "libzmq-v141-x64-4_3_2" (
	powershell -Command "(New-Object Net.WebClient).DownloadFile('https://dl.bintray.com/zeromq/generic/libzmq-v141-x64-4_3_2.zip', 'libzmq-v141-x64-4_3_2.zip')"
	powershell Expand-Archive libzmq-v141-x64-4_3_2.zip -DestinationPath libzmq-v141-x64-4_3_2
)

if NOT exist "libzmq-v141-4_3_2" (
	powershell -Command "(New-Object Net.WebClient).DownloadFile('https://dl.bintray.com/zeromq/generic/libzmq-v141-4_3_2.zip','libzmq-v141-4_3_2.zip')"
	powershell Expand-Archive libzmq-v141-4_3_2.zip -DestinationPath libzmq-v141-4_3_2
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