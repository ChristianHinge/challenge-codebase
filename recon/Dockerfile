FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-all-dev \
    libhdf5-dev \
    python3-dev \
    python3-pip \
    swig \
    libfftw3-dev
RUN apt-get install libinsighttoolkit5-dev -y
RUN apt install nlohmann-json3-dev -y

# Clone and build STIR
RUN git clone https://github.com/UCL/STIR
RUN pip install numpy
RUN mkdir build && cd build && cmake ../STIR -DBUILD_SWIG_PYTHON=ON -DSTIR_BUILD_EXECUTABLES=ON -DPython_EXECUTABLE=/usr/bin/python3 && make -j 8
RUN pip install nibabel