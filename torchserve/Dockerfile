FROM pytorch/torchserve:latest

USER root

RUN pip install --no-cache-dir transformers numpy sentencepiece

WORKDIR /home/model-server

COPY torchserve/model-store /home/model-server/model-store
COPY torchserve/config.properties /home/model-server/config.properties

USER model-server


CMD ["torchserve", "--start", "--ts-config", "/home/model-server/config.properties", "--models", "mymodel=mymodel.mar", "--foreground"]
