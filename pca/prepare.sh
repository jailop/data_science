#!/bin/sh

cat data/in/winequality-white.csv \
    | tail -n +2 \
    | sed 's/;/\t /g' \
    > data/tmp/winequality-white.csv
