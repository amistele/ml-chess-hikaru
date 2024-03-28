FROM continuumio/miniconda3:latest

# SET WORK DIRECTORY
WORKDIR /opt/app

# USE BASH SHELL FOR DEFAULT
SHELL ["/bin/bash","--login","-c"]

# SET UP CONDA ENV
RUN conda init bash
RUN conda install -y python=3.11
RUN conda install -y numpy
RUN conda install -y pandas
RUN conda install -y nvidia/label/cuda-12.1.0:cuda-toolkit -c pytorch -c nvidia
RUN conda install -y pytorch-cuda=12.1 -c pytorch -c cuda
RUN conda install -y pytorch -c pytorch
RUN conda install -y jupyter
RUN conda install -y ipython
# ONE PIP INSTALL TOO
RUN pip install chess

# COPY NECESSARY FILES AND LIBRARIES IN
COPY ./chessfun.py chessfun.py
COPY ./SmoothBrainChess.py SmoothBrainChess.py
COPY ./model_weights_2k_3lin_do.pth model_weights_2k_3lin_do.pth
COPY ./PLAYCHESS.ipynb PLAYCHESS.ipynb
COPY ./Output_Node_Key.csv Output_Node_Key.csv

# EXPOSE A PORT FOR JUPYTER NOTEBOOK
EXPOSE 8888

# RUN THE NOTEBOOK!
CMD ["jupyter","notebook","--ip='0.0.0.0'","--port=8888","--no-browser","--allow-root","--NotebookApp.token=''","--NotebookApp.password=''"]
