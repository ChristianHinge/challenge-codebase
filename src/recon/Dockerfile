# ── Stage 1: Build STIR ──────────────────────────────────────────────────────
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential \
    libboost-all-dev libhdf5-dev libfftw3-dev \
    python3-dev python3-pip \
    swig libinsighttoolkit5-dev nlohmann-json3-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy

RUN git clone https://github.com/UCL/STIR \
 && cd STIR \
 && git checkout 46da4ef2b8b38487b08f5aa48c6e38b568a584cc

RUN mkdir /build && cd /build \
 && cmake /STIR \
    -DBUILD_SWIG_PYTHON=ON \
    -DSTIR_BUILD_EXECUTABLES=ON \
    -DPython_EXECUTABLE=/usr/bin/python3 \
 && make -j$(nproc) \
 && make install

RUN pip install --no-cache-dir nibabel scipy

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libboost-filesystem1.74.0 \
    libboost-program-options1.74.0 \
    libboost-regex1.74.0 \
    libboost-serialization1.74.0 \
    libboost-thread1.74.0 \
    libboost-iostreams1.74.0 \
    libboost-date-time1.74.0 \
    libhdf5-103 \
    libfftw3-double3 \
 && rm -rf /var/lib/apt/lists/*

# Copy STIR install (executables, libs, Python bindings) + pip packages
COPY --from=builder /usr/local /usr/local
RUN ldconfig

WORKDIR /recon
COPY . .

ENV PYTHONPATH="/usr/local/python"
ENV OVERWRITE=""

CMD python3 main.py \
    /data/recon \
    /data/ct/ct.nii.gz \
    /data/output/pet.nii.gz \
    --intermediates_dir /data/intermediates \
    ${OVERWRITE:+--overwrite}
