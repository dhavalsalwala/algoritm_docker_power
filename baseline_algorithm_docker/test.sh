#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh


VOLUME_SUFFIX="VOL"
#$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="15g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create baseline_algorithm_docker-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/:/input/ \
        -v baseline_algorithm_docker-output-$VOLUME_SUFFIX:/output/ \
        baseline_algorithm_docker

docker run --rm \
        -v baseline_algorithm_docker-output-$VOLUME_SUFFIX:/output/ \
        python:3.9-slim cat /output/vessel-power-estimates.json | python3 -m json.tool

docker run --rm \
        -v baseline_algorithm_docker-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        python:3.9-slim python3 -c "import json, sys; f1 = json.load(open('/output/vessel-power-estimates.json')); sys.exit(True);"

if [ $? ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm baseline_algorithm_docker-output-$VOLUME_SUFFIX
