ARG DOCKER_PLATFORM=linux/amd64
FROM --platform=${DOCKER_PLATFORM} nvidia/cuda:11.4.3-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CUTENSOR_VERSION=1.3.3.2
ARG CUTENSOR_ARCHIVE=libcutensor-linux-x86_64-${CUTENSOR_VERSION}-archive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        make \
        wget \
        xz-utils && \
    wget -q https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/${CUTENSOR_ARCHIVE}.tar.xz && \
    tar -xJf ${CUTENSOR_ARCHIVE}.tar.xz -C /opt && \
    mv /opt/${CUTENSOR_ARCHIVE} /opt/cutensor && \
    rm -f ${CUTENSOR_ARCHIVE}.tar.xz && \
    rm -rf /var/lib/apt/lists/*

ENV CUTENSOR_HOME=/opt/cutensor
ENV LD_LIBRARY_PATH=/opt/cutensor/lib/11:${LD_LIBRARY_PATH}

WORKDIR /workspace

CMD ["make"]
