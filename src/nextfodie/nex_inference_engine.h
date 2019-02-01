/*
 *******************************************************************************
 *
 * Copyright (C) 2019 NEXAIOT Co., Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************
 */
#pragma once
#include <iostream>
#include <string>
#include <vector>

#include <cpprest/json.h>
#include <opencv2/opencv.hpp>

#include <inference_engine.hpp>

using namespace InferenceEngine;
using namespace web;

namespace NexInferenceEngine {

void display_intel_ie_version();

class ObjectDetection {
private:
    float threshold;

    int input_w;
    int input_h;
    int input_ch;
    std::string output_type;

    InferencePlugin plugin;
    ExecutableNetwork network;
    InferRequest infer_request;
    Blob::Ptr input_blob;
    uint8_t* blob_data;
    int object_size;
    int max_output_count;

    std::string validateNetwork(CNNNetReader &reader);
    std::string findPluginPath();
    void loadPlugin(std::string &app_path, std::string &device);

public:
    ObjectDetection(std::string &app_path, std::string &device);
    ObjectDetection(std::string &app_path, std::string &device, std::string &model_xml, float threshold=0.5);

    void loadModel(std::string &model_xml, std::string &model_bin);
    void setThreshold(float threshold) {this->threshold = threshold;};
    cv::Mat openImage(std::string imagepath) {return cv::imread(imagepath);};
    cv::Mat openImage(std::vector<char> raw_data) {return cv::imdecode(cv::Mat(raw_data), cv::IMREAD_COLOR);};
    cv::Mat openImage(char *raw_data, size_t size) {
        std::vector<char> vec(raw_data, raw_data + size);
        return cv::imdecode(cv::Mat(vec), cv::IMREAD_COLOR);
    };
    const float* infer(cv::Mat &img);
    json::value parse(const float *detections, bool normalized=true, float threshold=-1);
};

} // namespace NexInferenceEngine