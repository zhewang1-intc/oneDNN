source /home/zhe/intel/oneapi/setvars.sh
mkdir -p build
cd build

export CC=icx
export CXX=icpx

cmake .. \
          -DDNNL_CPU_RUNTIME=SYCL \
          -DDNNL_GPU_RUNTIME=SYCL \

make -j