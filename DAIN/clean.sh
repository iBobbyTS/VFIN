#!/usr/bin/env bash

cd my_package

cd MinDepthFlowProjection
rm -rf build *.egg-info dist
cd ..

cd FlowProjection
rm -rf build *.egg-info dist
cd ..

cd SeparableConv
rm -rf build *.egg-info dist
cd ..

cd InterpolationCh
rm -rf build *.egg-info dist
cd ..

cd DepthFlowProjection
rm -rf build *.egg-info dist
cd ..

cd Interpolation
rm -rf build *.egg-info dist
cd ..

cd SeparableConvFlow
rm -rf build *.egg-info dist
cd ..

cd FilterInterpolation
rm -rf build *.egg-info dist
cd ..

cd ..
cd PWCNet/correlation_package_pytorch1_0
rm -rf build *.egg-info dist
cd ../..
