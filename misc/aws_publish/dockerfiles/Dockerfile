# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

ENV PYTHONUNBUFFERED TRUE

ARG tarball
ARG requirements
ARG my_secret
ARG task_code
ARG model_store
ARG model_name

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh

RUN mkdir -p /home/model-server/ && mkdir -p /home/model-server/tmp
COPY config.properties /home/model-server/config.properties
COPY ${task_code}.json /home/model-server/code/

ENV TEMP=/home/model-server/tmp
ADD ${tarball} /home/model-server/code

RUN ls /home/model-server/code

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    fakeroot \
    ca-certificates \
    dpkg-dev \
    g++ \
    python3.8-dev \
    python3-pip \
    openjdk-11-jre-headless \
    curl \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN cd /tmp \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py

RUN python -m pip install --no-cache-dir torch==1.12.1+cu113 torchaudio==0.12.1+cu113 torchserve -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /home/model-server/code

RUN if [ ${requirements} = True ]; then python -m pip install --no-cache-dir --force-reinstall -r requirements.txt; fi

RUN python -c 'import fairseq; print(fairseq.__path__)'
RUN python -c 'import handler'

ENV PYTHONIOENCODING=UTF-8
ENV MODEL_STORE=${model_store}
ENV MODEL_NAME=${model_name}
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]
