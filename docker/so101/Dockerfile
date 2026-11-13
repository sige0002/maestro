ARG ROS_IMAGE=ros:jazzy-ros-base
FROM ${ROS_IMAGE}
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-pip ros-jazzy-rmw-fastrtps-cpp ros-jazzy-tf2-ros-py libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv --system-site-packages /opt/maestro
ENV PATH=/opt/maestro/bin:$PATH
COPY docker/so101/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /maestro
COPY maestro /maestro/maestro
COPY config /maestro/config
COPY docker/so101/entrypoint.sh /entrypoint-maestro.sh
ENTRYPOINT ["/bin/bash", "/entrypoint-maestro.sh"]
CMD ["--check"]
