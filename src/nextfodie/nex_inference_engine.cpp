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
#include <cstdlib>          // std::getenv
#include <memory>
#include <stdexcept>
//#include <sys/utsname.h>

#include <ext_list.hpp>     // InferenceEngine::Extensions::Cpu::CpuExtensions

#include "nex_inference_engine.h"

static std::string model_bin_filename(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath + ".bin";
    return filepath.substr(0, pos) + ".bin";
}

namespace NexInferenceEngine {

void display_intel_ie_version() {
    const Version *ver = GetInferenceEngineVersion();
    std::cout << "OpenVINO Inference Engine API " << ver->apiVersion.major << "." << ver->apiVersion.minor
              << " (Build " << (char*)ver->buildNumber << ")" << std::endl;
}

ObjectDetection::ObjectDetection(std::string &app_path, std::string &device) {
    this->plugin = PluginDispatcher({this->findPluginPath(), ""}).getPluginByDevice(device);
    if (device.find("CPU") != std::string::npos) {
        auto ext_path = app_path + "lib/libcpu_extension.so";
        IExtensionPtr extension_ptr = make_so_pointer<IExtension>(ext_path);
        this->plugin.AddExtension(extension_ptr);
    }

    this->input_w  = 0;
    this->input_h  = 0;
    this->input_ch = 0;
    this->setThreshold(0.5);
}

ObjectDetection::ObjectDetection(std::string &app_path, std::string &device, std::string &model_xml, float threshold) {
    this->plugin = PluginDispatcher({this->findPluginPath(), ""}).getPluginByDevice(device);
    if (device.find("CPU") != std::string::npos) {
        auto ext_path = app_path + "lib/libcpu_extension.so";
        IExtensionPtr extension_ptr = make_so_pointer<IExtension>(ext_path);
        this->plugin.AddExtension(extension_ptr);
    }

    std::string model_bin = model_bin_filename(model_xml);
    this->loadModel(model_xml, model_bin);
    this->setThreshold(threshold);
}

std::string ObjectDetection::findPluginPath() {
    std::string intel_cvsdk_dir = std::string(std::getenv("INTEL_CVSDK_DIR"));
    std::string plugin_path = intel_cvsdk_dir + "deployment_tools/inference_engine/lib/intel64";
    return plugin_path;
}

std::string ObjectDetection::validateNetwork(CNNNetReader &reader) {
    // Validate network input
    // SSD-based network should have one input and one output
    // https://software.intel.com/en-us/articles/OpenVINO-InferEngine #Understanding Inference Engine Memory Primitives
    InputsDataMap input_info(reader.getNetwork().getInputsInfo());
    if (input_info.size() != 1) {
        throw std::logic_error("Network is not single input");
    }

    InputInfo::Ptr &input = input_info.begin()->second;
    std::string input_type = input_info.begin()->first;
    input->setPrecision(Precision::U8);
    input->getInputData()->setLayout(Layout::NCHW);

    // Validate network output
    OutputsDataMap output_info(reader.getNetwork().getOutputsInfo());
    if (output_info.size() != 1) {
        throw std::logic_error("Network is not single output");
    }

    DataPtr &output = output_info.begin()->second;
    this->output_type = output_info.begin()->first;

    const SizeVector output_dims = output->getTensorDesc().getDims();
    this->max_output_count = output_dims[2];
    this->object_size = output_dims[3];
    if (output_dims.size() != 4) {
        throw std::logic_error("Incorrect output dimensions for SSD");
    }
    if (this->object_size != 7) {
        throw std::logic_error("Output should have 7 as a last dimension");
    }
    output->setPrecision(Precision::FP32);
    output->setLayout(Layout::NCHW);

    return input_type;
}

void ObjectDetection::loadModel(std::string &model_xml, std::string &model_bin) {
    CNNNetReader reader;

    reader.ReadNetwork(model_xml);
    reader.getNetwork().setBatchSize(1);
    reader.ReadWeights(model_bin);

    auto input_type = this->validateNetwork(reader);
    this->network = this->plugin.LoadNetwork(reader.getNetwork(), {});
    this->infer_request = this->network.CreateInferRequest();
    this->input_blob = this->infer_request.GetBlob(input_type);
    auto blob_size = this->input_blob->getTensorDesc().getDims();
    this->input_w  = (int)blob_size[3];
    this->input_h  = (int)blob_size[2];
    this->input_ch = (int)blob_size[1];
}

const float* ObjectDetection::infer(cv::Mat &img) {
    if (img.empty()) {
        throw std::logic_error("Failed to get frame from image file");
    }
    int img_w = img.size().width;
    int img_h = img.size().height;

    // Resize and keep aspect ratio
    if (this->input_w != img_w || this->input_h != img_h) {
        cv::Mat img_new;
        double ratio_w = (double)this->input_w / (double)img_w;
        double ratio_h = (double)this->input_w / (double)img_h;
        double ratio = (ratio_w < ratio_h)? ratio_w : ratio_h;
        int interpolation = (ratio < 1.0)? cv::INTER_AREA : cv::INTER_CUBIC;
        cv::resize(img, img_new, cv::Size(), ratio, ratio, interpolation);

        int resized_w = img_new.size().width;
        int resized_h = img_new.size().height;
        int top = 0;
        int bottom = 0;
        int left = 0;
        int right = 0;
        if (resized_w < this->input_w) {
            left = (this->input_w - resized_w) / 2;
            right = this->input_w - left;
        }
        if (resized_h < this->input_h) {
            top = (this->input_h - resized_h) / 2;
            bottom = this->input_h - top;
        }
        if (top + left > 0) {
            img = img_new;
            cv::copyMakeBorder(img, img_new, top, bottom, left, right, cv::BORDER_CONSTANT);
            img = img_new;
        }
    }

    // place resized image data into blob
    uint8_t* blob_data = static_cast<uint8_t*>(this->input_blob->buffer());
    for (size_t c = 0; c < this->input_ch; c++) {
        for (size_t h = 0; h < this->input_h; h++) {
            for (size_t w = 0; w < this->input_w; w++) {
                blob_data[c * this->input_w * this->input_h + h * this->input_h + w] = img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }

    // Do infer
    this->infer_request.Infer();

    return this->infer_request.GetBlob(this->output_type)->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
}

json::value ObjectDetection::parse(const float *detections, bool normalized, float threshold) {
    float th = (threshold < 0)? this->threshold : threshold;
    std::vector<json::value> objs;
    for (int idx = 0; idx < this->max_output_count; idx++) {
        if (detections[idx * this->object_size + 0] < 0) {
            break;
        }

        float score = detections[idx * this->object_size + 2];
        if (score < th) {
            continue;
        }

        json::value obj;
        obj["class_id"] = json::value::number(static_cast<int>(detections[idx * this->object_size + 1]));
        obj["class"]    = json::value::string("");
        obj["score"]    = json::value::number(score);
        obj["bbox"]     = json::value::array(4);
        if (normalized) {
            obj["bbox"][0] = json::value::number(detections[idx * this->object_size + 3]);   // xmin
            obj["bbox"][1] = json::value::number(detections[idx * this->object_size + 4]);   // ymin
            obj["bbox"][2] = json::value::number(detections[idx * this->object_size + 5]);   // xmax
            obj["bbox"][3] = json::value::number(detections[idx * this->object_size + 6]);   // ymax
        } else {
            obj["bbox"][0] = json::value::number((int)(detections[idx * this->object_size + 3] * this->input_w));  // xmin
            obj["bbox"][1] = json::value::number((int)(detections[idx * this->object_size + 4] * this->input_h));  // ymin
            obj["bbox"][2] = json::value::number((int)(detections[idx * this->object_size + 5] * this->input_w));  // xmax
            obj["bbox"][3] = json::value::number((int)(detections[idx * this->object_size + 6] * this->input_h));  // ymax
        }
        objs.push_back(obj);
    }

    return json::value::array(objs);
}

}; // namespace NexInferenceEngine