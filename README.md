# `nextfodie`

## Introduction
This is a RESTful API server using [Microsoft C++ REST SDK](https://github.com/Microsoft/cpprestsdk) and [MPFDParser](http://grigory.info/MPFDParser.About.html) to perform inference based on [Intel OpenVINO 2018 R5](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux). The model is trained by TensorFlow Object Detection API, then use Model Optimizer of OpenVINO to convert to IR format.
The default port number is `30303`.

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

### Install Microsoft C++ REST SDK
For detailed steps to install cpprestsdk, please follow [here](https://github.com/Microsoft/cpprestsdk) or [apt-get](https://launchpad.net/ubuntu/+source/casablanca/2.8.0-2build2) on Debian/Ubuntu

``` bash
$ sudo apt-get install libcpprest-dev
```

## Build `nextfodie`

Clone this repository.
``` bash
$ cd ~
$ git clone https://github.com/nexgus/nextfodie.git
```

Make sure the [CMake](https://cmake.org/) is istalled. To build `nextfodie`, the minimum version should be 2.8. Use this command to install the latest `CMake` on Ubuntu.
``` bash
$ sudo apt-get install cmake
```

Build it!
``` bash
$ cd ~/nextfodie/src
$ bash build.sh
```

## Run `nextfodie`
The generated binary in `~/nextfodie/build/intel64/Release`. So move to there.
``` bash
$ cd ~/nextfodie/build/intel64/Release
```

Run `nextfodie` with CPU, load a pre-trained model, listen to localhost
``` bash
$ ./nextfodie -m ir/fp32/frozen_inference_graph.xml
Loading model... done
Listen to http://localhost:30303
```

Run `nextfodie` with GPU, do not load model, listen to anyone
``` bash
$ ./nextfodie -d GPU -H 0.0.0.0
Listen to http://0.0.0.0:30303
```
Now you may use any RESTful API tool such as [Postman](https://www.getpostman.com/) or [`curl`](https://curl.haxx.se/) to load model. The response should be like this
```
---------- PUT /model
Load model request (xml: 173209; bin: 94699118)
Loading... done (rx: 8017.15mS; load: 15598.1mS; total: 23615.2mS)
```

## Build `nextfodie` in Docker
You may refer to [openvino-docker](https://github.com/mateoguzman/openvino-docker) to build your own Docker image or using `Dockerfile.16.04` or `Dockerfile.18.04` directlly.

Suppose your cloned `nextfodie` repository is in `~/nextfodie`

For Ubuntu 16.04
``` bash
$ cd ~/nextfodie/docker
$ docker build -t nextfodie:16.04 -f Dockerfile.16.04 .
$ cd ~
$ docker run --rm -it -v $(pwd)/nextfodie:/root/nextfodie -p 30303:30303 nextfodie:16.04
```

For Ubuntu 18.04
``` bash
$ cd ~/nextfodie/docker
$ docker build -t nextfodie:18.04 -f Dockerfile.18.04 .
$ cd ~
$ docker run --rm -it -v $(pwd)/nextfodie:/root/nextfodie -p 30303:30303 nextfodie:18.04
```

## Run `nextfodie` with GPU in Docker
You have to add device `/dev/dri` into container. For example,
``` bash
$ docker run --rm -it --device /dev/dri -v $(pwd)/nextfodie:/root/nextfodie -p 30303:30303 nextfodie:18.04
```