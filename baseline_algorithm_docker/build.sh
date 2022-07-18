#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t baseline_algorithm_docker "$SCRIPTPATH"
