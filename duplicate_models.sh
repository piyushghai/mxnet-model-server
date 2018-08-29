#!/usr/bin/env bash

# Script to duplicate files in a foler. This script appends a number as a suffix to the file.
# Example : cp -a a/ a-1/
set -ex
cd $1
echo `pwd`
for ((i=1; i <= $3; i++))
do 
	cp -a $2/ $2-$i
done
