echo "############################# initial cleanup ############################# "
# cleanup wrapper
rm -r build
rm -r lib

echo "#############################  build starspace ############################# "
# build starspace lib
cd ..
make clean
make -f makefile_py
cd -

echo "#############################  build wrapper ############################# "
# build wrapper
mkdir lib
cp ../libstarspace.a ./lib
mkdir build
cd build
echo "Conan install"
conan install ..
echo "cmake .."
cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_LIBRARY=/Users/magdiel/miniconda3/envs/python36/lib/libpython3.6m.dylib -DPYTHON_INCLUDE_DIR=/Users/magdiel/miniconda3/envs/python36/include/python3.6m/
echo "cmake build"
cmake --build .
cd -

echo "#############################  run test ############################# "
# run test
# this will run all wrapped APIs available at this moment.
# by loading traing data from input.txt, train with train mode 5, 
# find nearest neighbor to some random text, save model as binary and tsv,
# try loading both saved models above again and 
# generate Document Embedding for some random text.
cp ./build/starwrap.so ./test
cd test
python3 test.py
cd -