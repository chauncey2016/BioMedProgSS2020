#!/bin/bash

python3 training.py -m resnet -o adam -e 200 -i 224 -c 3
python3 training.py -m densenet -o adam -e 200 -i 224 -c 3
python3 training.py -m inceptionv4 -o adam -e 200 -i 224 -c 3
python3 training.py -m resnet -o sgd -e 200 -i 224 -c 3
python3 training.py -m densenet -o sgd -e 200 -i 224 -c 3
python3 training.py -m inceptionv4 -o sgd -e 200 -i 224 -c 3
