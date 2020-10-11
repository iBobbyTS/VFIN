#!/usr/bin/env bash

if [ -z "$1" ]
then
	python_executable=python
else
	python_executable=$1
fi

if [ ! -f "$python_executable" ]
then
	python_executable=python
fi

echo "Building CUDAExtension for PyTorch in $python_executable"
echo "You need torch>=1.0.0, <=1.4.0, you have $($python_executable -c "import torch; print(torch.__version__)")"

cd my_package
export PYTHONPATH=$python_executable:$(pwd)

cd MinDepthFlowProjection
rm -rf build *.egg-info dist
$python_executable setup.py install
cd ..

cd FlowProjection
rm -rf build *.egg-info dist
$python_executable setup.py install
cd ..

cd SeparableConv
rm -rf build *.egg-info dist
$python_executable setup.py install
cd ..

cd InterpolationCh
rm -rf build *.egg-info dist
$python_executable setup.py install
cd ..

cd DepthFlowProjection
rm -rf build *.egg-info dist
$python_executable setup.py install
cd ..

cd Interpolation
rm -rf build *.egg-info dist
$python_executable setup.py install
cd ..

cd SeparableConvFlow
rm -rf build *.egg-info dist
$python_executable setup.py install
cd ..

cd FilterInterpolation
rm -rf build *.egg-info dist
$python_executable setup.py install
cd ..


cd ../PWCNet/correlation_package_pytorch1_0
rm -rf build *.egg-info dist
$python_executable setup.py install
cd ../..
