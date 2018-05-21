#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
from g2s import run as g2s

serverAddressList=['localhost'];
if len(sys.argv)>1 :
	file_name = sys.argv[1]
	fp = open(file_name)
	serverAddressList = fp.read().splitlines()

for serverAddress in serverAddressList:
	g2s('-sa',serverAddress,'-shutdown')
