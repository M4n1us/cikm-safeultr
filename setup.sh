#!/bin/bash

conda create --name safeultr
conda config --add channels pytorch nvidia conda-forge
conda install --yes --file requirements.txt