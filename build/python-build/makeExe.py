import os
import urllib.request
import subprocess
import zipfile


if not(os.path.exists('C:\Program Files\ZeroMQ 4.0.4')):
	urllib.request.urlretrieve("https://miru.hk/archive/ZeroMQ-4.0.4~miru1.0-x64.exe", "ZeroMQ-4.0.4~miru1.0-x64.exe");
	subprocess.call('ZeroMQ-4.0.4~miru1.0-x64.exe');

if not(os.path.exists('cppzmq-master')):
	urllib.request.urlretrieve("https://codeload.github.com/zeromq/cppzmq/zip/master", "cppzmq-master.zip");
	with zipfile.ZipFile("cppzmq-master.zip","r") as zip_ref:
		zip_ref.extractall(".")

if not(os.path.exists('jsoncpp-master')):
	urllib.request.urlretrieve("https://codeload.github.com/open-source-parsers/jsoncpp/zip/master", "jsoncpp-master.zip");
	with zipfile.ZipFile("jsoncpp-master.zip","r") as zip_ref:
		zip_ref.extractall(".")
	subprocess.Popen("python amalgamate.py", cwd="./jsoncpp-master")

subprocess.Popen("python setup_Win.py bdist --format=wininst");
