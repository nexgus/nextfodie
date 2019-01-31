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
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <sys/stat.h>

#include <cpprest/http_listener.h>
#include <gflags/gflags.h>

#include "nex_inference_engine.h"
#include "nex_request_handler.h"

using namespace web::http::experimental::listener;
namespace NexIE = NexInferenceEngine;

NexIE::ObjectDetection *ie = NULL;

static const char help_message[] = "Display this help and exit";
static const char host_message[] = "Host name/IP (default: localhost)";
static const char port_message[] = "Port number (default 30303)";
static const char device_message[] = "Specify the target device to infer on (default: CPU); CPU and GPU is acceptable.";
static const char model_message[] = "Path to an .xml file with a trained model";
static const char threshold_message[] = "Threshold for inference score/probability (default: 0.5)";

DEFINE_bool  (h, false,       help_message);
DEFINE_string(H, "localhost", host_message);
DEFINE_int32 (p, 30303,       port_message);
DEFINE_string(d, "CPU",       device_message);
DEFINE_string(m, "",          model_message);
DEFINE_double(t, 0.7,         threshold_message);

static void show_usage() {
    std::cout << std::endl;
    std::cout << "nextfodie [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h              " << help_message << std::endl;
    std::cout << "    -H <string>     " << host_message << std::endl;
    std::cout << "    -p <integer>    " << port_message << std::endl;
    std::cout << "    -d <string>     " << device_message << std::endl;
    std::cout << "    -m <string>     " << model_message << std::endl;
    std::cout << "    -t <double>     " << threshold_message << std::endl;
    std::cout << std::endl;
    NexIE::display_intel_ie_version();
    std::cout << std::endl;
}

static bool parse_cli(int argc, char *argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        show_usage();
        return false;
    }
    if ((FLAGS_p < 1) || (FLAGS_p > 65535)) {
        throw std::logic_error("Parameter -p should be in range 1-65535 (default: 30303)");
    }
    if ((FLAGS_t < 0) || (FLAGS_t > 1)) {
        throw std::logic_error("Parameter -t must be between 0 and 1 (default: 0.7)");
    }
    if ((FLAGS_d != "CPU") && (FLAGS_d != "GPU")) {
        throw std::logic_error("Parameter -d must be CPU or GPU");
    }
    return true;
}

static std::string& find_application_path(char *argv[]) {
    static std::string app_path;
    std::string command = std::string(argv[0]);
    size_t pos = command.rfind('/');
    if (pos == std::string::npos) {
        std::string paths = std::string(std::getenv("PATH"));
        size_t start_pos = 0;
        struct stat buffer;
        while (true) {
            std::string path;
            pos = paths.find(':');
            if (pos == std::string::npos) {
                app_path = paths.substr(start_pos);
            }
            else {
                app_path = paths.substr(start_pos, pos);
                start_pos = pos + 2;
            }
            auto filepath = app_path + '/' + command;
            if (stat(filepath.c_str(), &buffer) == 0) {
                break;
            }
        }
    }
    else {
        app_path = command.substr(0, pos+1);
    }
    return app_path;
}

int main(int argc, char *argv[]) {
    if (!parse_cli(argc, argv)) {
        return 0;
    }

    auto app_path = find_application_path(argv);
    if (FLAGS_m.size() > 0) {
        std::cout << "Loading model...";
        ie = new NexIE::ObjectDetection(app_path, FLAGS_d, FLAGS_m);
        std::cout << " done" << std::endl;
    }
    else {
        ie = new NexIE::ObjectDetection(app_path, FLAGS_d);
    }
    ie->setThreshold(FLAGS_t);

    std::string addr = FLAGS_H + ":" + std::to_string(FLAGS_p);
    if (FLAGS_H.find("://") == std::string::npos) {
        addr = "http://" + addr;
    }
    http_listener listener(addr);

    listener.support(methods::GET,  handle_get);
    listener.support(methods::POST, handle_post);
    listener.support(methods::PUT,  handle_put);
    //listener.support(methods::DEL,  handle_del);

    std::cout << "Listen to " << addr << std::endl;
    try {
        listener.open()
                .then([&listener]() {})
                .wait();
        while (true);
    }
    catch (std::exception const &e) {
        std::cout << e.what() << std::endl;
        delete ie;
    }

    return 0;
}
