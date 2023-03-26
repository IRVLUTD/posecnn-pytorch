cd lib/layers/;
python setup.py build develop;
cd ../utils;
python setup.py build_ext --inplace;
cd ../../ycb_render;
python setup.py develop

