FROM  tensorflow/tensorflow:2.0.0rc0-gpu-py3

RUN apt-get update && \
    apt-get install -y git curl wget vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade --no-cache-dir --no-cache pip pandas

# keras-contrib
RUN cd /opt && \
    git clone https://www.github.com/keras-team/keras-contrib.git && \
    cd keras-contrib && \
    python convert_to_tf_keras.py && \
    USE_TF_KERAS=1 python setup.py install

WORKDIR /project
CMD ["bash"]


