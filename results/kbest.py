#!/usr/bin/python3
import unicodedata
import string
import re
from os import listdir
from os.path import isfile, join
import sys


	
file_object  = open("kbest_input.txt", "r")

texto=file_object.read()

lines=re.split(r'\n',texto,maxsplit=0,flags=0)
intlines=[]

for x in range (0,8):
	intlines.append(float(lines[x]))
	intlines.sort()

#>>>>Parte que interessa quando tiveres uma lista ordenanda de forma crescente<<<<

for x in range (0,6):
	

	if ( (intlines[x+1]<=intlines[x]*1.05) & (intlines[x+2]<=intlines[x+1]*1.05) ):
		a=(intlines[x]+intlines[x+1]+intlines[x+2])/3
		print(intlines[x])
		print(intlines[x+1])
		print(intlines[x+2])
		print("k-best= "+str(a))
		break		
file_object.close()

