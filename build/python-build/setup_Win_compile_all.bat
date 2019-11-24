if NOT exist "C:\Program Files\ZeroMQ 4.0.4" (
	powershell -Command "(New-Object Net.WebClient).DownloadFile('https://miru.hk/archive/ZeroMQ-4.0.4~miru1.0-x64.exe', 'ZeroMQ-4.0.4~miru1.0-x64.exe')"
	start /w ZeroMQ-4.0.4~miru1.0-x64.exe
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

for %%x in ( 3.8, 3.7, 3.6, 3.5) do (
	py -%%x -m pip install --upgrade pip
	py -%%x -m pip install numpy setuptools wheel
	py -%%x setup.py bdist --format=wininst
	del build /s /f /q
)