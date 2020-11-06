#!/bin/bash
#echo "Compiling [create_bi_grid]..."
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

echo $TF_CFLAGS
echo $TF_LFLAGS
nvcc -std=c++11 -c -o outlier.cu.o outlier.cu.cc \
	-I /usr/local \
  	${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -DNDEBUG 

g++ -std=c++11 -shared -o outlier.so outlier.cc \
	-L /usr/local/cuda-9.0/lib64/ \
	outlier.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} 

rm outlier.cu.o