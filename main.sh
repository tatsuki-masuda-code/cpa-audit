#!/bin/bash
export PATH=$PATH:$HOME/.local/bin
pip3 install -qr requirements.txt
pip3 install -qe .
cpa-audit