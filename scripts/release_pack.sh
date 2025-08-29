#!/bin/bash
# Package model and config for release
mkdir -p release
cp configs/user_infer.yaml release/
# In real use, copy trained model as well
