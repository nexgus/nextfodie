# Welcome!
This is a RESTful API server to perform inference based on Intel OpenVINO 2018 R5.

## API
```
GET /inference
POST /inference
PUT /model
```
For detail usage, please check [source code](https://github.com/nexgus/nextfodie/blob/master/src/nextfodie/nex_request_handler.cpp)

## Dependencies
* [openvino](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux)
* [cpprestsdk](https://github.com/Microsoft/cpprestsdk)
* [gflags](https://github.com/gflags/gflags)
* [MPFDParser](http://grigory.info/MPFDParser.About.html)

### Install Intel OpenVINO
For detailed steps to install OpenVINO, follow [Install the Intel Distribution of OpenVINO toolkit for Linux](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux).

### Install Microsoft C++ RESR SDK
For detailed steps to install cpprestsdk, please follow [here](https://github.com/Microsoft/cpprestsdk) or [apt-get](https://launchpad.net/ubuntu/+source/casablanca/2.8.0-2build2) on Debian/Ubuntu

``` bash
$ sudo apt-get install libcpprest-dev
```

## Build nextfodie

Clone this repository.

``` bash
$ git clone https://github.com/nexgus/nextfodie.git
```

Make sure the [CMake](https://cmake.org/) is istalled. To build nextfodie, the minimum version should be 2.8. Use this command to install the latest CMake on Ubuntu.

``` bash
$ sudo apt-get install cmake
```

Build it!

``` bash
$ cd nextfodie/src
$ ./build.sh
```

Find the generated binary in this folder.
```
nextfodie/build/intel64/Release
```
