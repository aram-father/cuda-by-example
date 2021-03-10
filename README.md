# cuda-by-example

**Author:** Wonseok Lee (aram_fahter@naver.com)

**Last update:** 2021-03-09(TUE)

## What does this repository do?

This repository implements the examples illustrated in [CUDA By Example](https://developer.nvidia.com/cuda-example).

To make the examples more practical, I've modified the given examples slightly.

For example,
- All the examples in this repository are built using cmake
- Given utils(in the book) are replaced with more familiar ones such as OpenCV

## How to build & execute it?

For all the examples, I've given an index with two digits(i.e. `xx_{example_name}` where `xx` is an index with two digits).

Expalnation for each example could be found in `xx_{example_name}/README.md`.

To build and execute each example, follow below steps:

```bash
git clone https://github.com/aram-father/cuda-by-example.git
cd cuda-by-example
mkdir build
cd build
cmake ../xx_{example_name}
make
./xx_{example_name}
```

