TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

# robust warp
g++ -std=c++11 -shared robust_warp.cc -o robust_warp.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 -shared robust_warp_grad.cc -o robust_warp_grad.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# warp
g++ -std=c++11 -shared warp.cc -o warp.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 -shared warp_grad.cc -o warp_grad.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0