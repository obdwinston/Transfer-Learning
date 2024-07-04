#!/bin/bash

# run in terminal if you lack execution permission
# chmod +x run.sh

# uncomment the following to train model
python3 src/utils.py
python3 src/train.py

# change image url to predict different pokemon
url="https://oyster.ignimgs.com/mediawiki/apis.ign.com/pokemon-sun-pokemon-moon/6/6d/Partnercappikachu.jpg"

python3 src/predict.py "$url"
