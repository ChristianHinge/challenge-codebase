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
RUN git clone https://github.com/UCL/STIR && cd STIR && git checkout 46da4ef2b8b38487b08f5aa48c6e38b568a584cc


RUN pip install numpy
RUN mkdir build && cd build && cmake ../STIR -DBUILD_SWIG_PYTHON=ON -DSTIR_BUILD_EXECUTABLES=ON -DPython_EXECUTABLE=/usr/bin/python3 && make -j 8 && make install
RUN pip install nibabel scipy




# Copy recon folder
WORKDIR /recon
COPY . .

# Mount host directories to these paths at runtime:
#   -v /host/recon_folder:/data/recon
#   -v /host/ct.nii.gz:/data/ct/ct.nii.gz
#   -v /host/output:/data/output
#   -v /host/intermediates:/data/intermediates

# Set OVERWRITE=1 at runtime to enable overwriting existing files
ENV PYTHONPATH="/usr/local/python"
ENV OVERWRITE=""

CMD python3 main.py \
    /data/recon \
    /data/ct/ct.nii.gz \
    /data/output/pet.nii.gz \
    --intermediates_dir /data/intermediates \
    ${OVERWRITE:+--overwrite}
