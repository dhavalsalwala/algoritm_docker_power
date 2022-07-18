#!/usr/bin/env bash

./build.sh

docker save baseline_algorithm_docker | gzip -c > baseline_algorithm_docker.tar.gz
