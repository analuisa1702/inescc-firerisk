FROM --platform=linux/amd64 ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Lisbon

# --------------------------
# 1. Dependências base
# --------------------------
#RUN apt-get install -y software-properties-common && \
    #add-apt-repository universe && \
    #add-apt-repository main

RUN sed -i 's/^# deb/deb/g' /etc/apt/sources.list && \
    sed -i 's/^# deb-src/deb-src/g' /etc/apt/sources.list


RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    unzip \
    software-properties-common \
    make \
    gcc \
    libgcc-s1 \
    g++ \
    ccache \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-wxgtk4.0 \
    python3-numpy \
    python3-dateutil \
    python3-opengl \
    python3-gdal \
    gdal-bin \
    libgdal-dev \
    libproj-dev proj-data proj-bin \
    libgeos-dev \
    libsqlite3-dev sqlite3 \
    libtiff5-dev \
    libpnglite-dev \
    libcairo2 libcairo2-dev \
    libgsl-dev \
    libncurses-dev \
    zlib1g-dev libpq-dev \
    libreadline-dev \
    libfreetype-dev \
    libboost-thread-dev \
    libboost-program-options-dev \
    libzstd-dev gettext \
    libavformat-dev libavcodec-dev libswscale-dev \
    libfftw3-dev libbz2-dev ghostscript \
    libglu1-mesa-dev libxmu-dev \
    wx-common libxi-dev subversion \
    libwxgtk3.2-dev mesa-common-dev flex bison checkinstall \
    ffmpeg libavutil-dev libffmpegthumbnailer-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv --system-site-packages
ENV PATH="/venv/bin:$PATH"
ENV PYTHONPATH="/venv/lib/python3.12/site-packages"
RUN pip install --upgrade pip setuptools wheel

# --------------------------
# 2. Instalar GRASS GIS 8.4
# --------------------------
WORKDIR /tmp

RUN wget https://github.com/OSGeo/grass/archive/refs/tags/8.4.2.tar.gz && \
    tar -xzvf 8.4.2.tar.gz && \
    cd grass-8.4.2 && \
    export MYCFLAGS='-O2 -fPIC -fno-common -fexceptions -std=gnu99 -fstack-protector -m64' && \
    export MYLDFLAGS='-Wl,--no-undefined -Wl,-z,now' && \
    LDFLAGS="$MYLDFLAGS" CFLAGS="$MYCFLAGS" ./configure \
        --with-cxx \
        --enable-largefile \
        --with-proj --with-proj-share=/usr/share/proj \
        --with-gdal=/usr/bin/gdal-config \
        --with-python \
        --with-geos \
        --with-sqlite \
        --with-nls \
        --with-zstd \
        --with-cairo --with-cairo-ldflags=-lfontconfig \
        --with-freetype=yes --with-freetype-includes="/usr/include/freetype2/" \
        --with-wxwidgets \
        --with-fftw \
        --with-motif \
        --with-opengl-libs=/usr/include/GL \
        --without-postgres \
        --without-netcdf \
        --without-mysql \
        --without-odbc \
        --without-ffmpeg \
        --without-pdal \
        --without-openmp \
        && \
    make -j4 && \
    make install

# --------------------------
# 3. Variáveis de ambiente
# --------------------------
ENV GRASS_BIN=/usr/local/bin/grass
ENV GISBASE=/usr/local/grass84
ENV PATH="$PATH:${GISBASE}/bin:${GISBASE}/scripts"

ENV PYTHONPATH="${GISBASE}/etc/python:$PYTHONPATH"
ENV GRASS_PYTHON=/venv/bin/python3

ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_DATA=/usr/share/proj

# --------------------------
# 4. Setup GLASS
# --------------------------
RUN git clone https://github.com/jasp382/glass /glass
RUN cd /glass && git checkout dev26

RUN cd /glass && pip install -r requirements.txt --break-system-packages
#RUN cd /pymov && pip install -r requirements.txt
ENV PYTHONPATH="${PYTHONPATH:-}:${PYTHONPATH:+${PYTHONPATH}:}/glass"


# --------------------------
# 4. Workspace e Jupyter
# --------------------------
RUN mkdir -p /code
WORKDIR /code

RUN pip install --upgrade pip && \
    pip install jupyterlab

EXPOSE 8889 

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8889", "--no-browser", "--allow-root"]