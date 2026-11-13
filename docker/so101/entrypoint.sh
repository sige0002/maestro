#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
exec python -m maestro.so101.agent_runner "$@"
