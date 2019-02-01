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
#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <ostream>
#include <sstream>
#include <string>

#include <cpprest/containerstream.h>
#include <cpprest/http_listener.h>
#include <cpprest/json.h>
#include <MPFDParser-1.1.1/Parser.h>

#include "nex_inference_engine.h"
#include "nex_request_handler.h"

using namespace web;
namespace NexIE = NexInferenceEngine;

extern NexIE::ObjectDetection *ie;

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

void handle_get(http_request request) {
    http::status_code status = status_codes::OK;
    json::value jsn;
    auto uri = request.relative_uri();
    auto path = uri.path();
    std::cout << "---------- GET " << uri.to_string() << std::endl;

    auto query = uri.query();
    auto paths = http::uri::split_path(http::uri::decode(path));
    if ((paths.size() != 1) || (paths[0] != "inference")) {
        status = status_codes::NotFound;
        std::ostringstream stream;
        stream << "Path not found (" << path << ")";
        jsn["error"] = json::value::string(stream.str());
        std::cout << stream.str() << std::endl;
    }
    else {
        auto queries = http::uri::split_query(query);
        int possible_query_count = queries.size();
        if ((possible_query_count < 1) || (possible_query_count > 3)) {
            status = status_codes::BadRequest;
            jsn["error"] = json::value::string("Bad Request (invalid query count)");
            std::cout << "Bad Request (invalid query count)" << std::endl;
        }
        else {
            if (queries.find("path") == queries.end()) {
                status = status_codes::BadRequest;
                jsn["error"] = json::value::string("Bad Request (cannot find path)");
                std::cout << "Bad Request (cannot find path)" << std::endl;
            }
            else {
                std::string::size_type sz;
                auto  imgpath = queries["path"];
                float threshold = -1;   // use default of inference engine
                bool  abs = false;

                possible_query_count--;
                if ((possible_query_count > 0) && (queries.find("threshold") != queries.end())) {
                    threshold = stof(queries["threshold"], &sz);
                    possible_query_count--;
                    if ((threshold < 0) || (threshold > 1)) {
                        threshold = -1;
                    }
                }
                if ((possible_query_count > 0) && (queries.find("abs") != queries.end())) {
                    if (queries["abs"] == "true") {
                        abs = true;
                    }
                    possible_query_count--;
                }

                if (possible_query_count > 0) {
                    status = status_codes::BadRequest;
                    jsn["error"] = json::value::string("Bad Request (unknown query)");
                    std::cout << "Bad Request (unknown query)" << std::endl;
                }
                else {
                    std::cout << "Inference request (image: " << imgpath << "; threshold: " << threshold 
                              << "; normalized: " << !abs << ")" << std::endl;

                    std::cout << "Infering..." << std::flush;
                    auto t0 = std::chrono::high_resolution_clock::now();
                    auto img = ie->openImage(imgpath);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    auto inference = ie->infer(img);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    jsn = ie->parse(inference, !abs, threshold);
                    auto t3 = std::chrono::high_resolution_clock::now();
                    status = status_codes::OK;

                    ms t_load  = std::chrono::duration_cast<ms>(t1 - t0);
                    ms t_infer = std::chrono::duration_cast<ms>(t2 - t1);
                    ms t_parse = std::chrono::duration_cast<ms>(t3 - t2);
                    ms t_total = std::chrono::duration_cast<ms>(t3 - t0);
                    std::cout << " done (load: " << t_load.count() << "mS; infer: " << t_infer.count() 
                              << "mS; parse: " << t_parse.count() << "mS; total: " << t_total.count() 
                              << "mS)" << std::endl;
                }
            }
        }
    }
    request.reply(status, jsn);
}

void handle_post(http_request request) {
    auto t0 = std::chrono::high_resolution_clock::now();
    http::status_code status = status_codes::OK;
    json::value jsn;
    auto uri = request.relative_uri();
    auto path = uri.path();
    std::cout << "---------- POST " << uri.to_string() << std::endl;

    auto paths = http::uri::split_path(http::uri::decode(path));
    if ((paths.size() != 1) || (paths[0] != "inference")) {
        status = status_codes::NotFound;
        std::ostringstream stream;
        stream << "Path not found (" << path << ")";
        jsn["error"] = json::value::string(stream.str());
        std::cout << stream.str() << std::endl;
    }
    else {
        http_headers headers = request.headers();
        concurrency::streams::istream body = request.body();

        char *img = NULL;
        unsigned long img_size = 0;
        double threshold = -1;  // use default of inference engine
        bool abs = false;
        std::string img_filename;

        if (headers.has("content-type")) {
            auto parser = MPFD::Parser();
            try {
                parser.SetUploadedFilesStorage(MPFD::Parser::StoreUploadedFilesInMemory);
                parser.SetMaxCollectedDataLength(std::numeric_limits<long>::max());
                parser.SetContentType(headers["content-type"]);

                // Move all body to parser to parse
                // ref: https://docs.microsoft.com/zh-tw/previous-versions/jj950083%28v%3dvs.140%29
                size_t total_read = 0;
                auto content_length = headers.content_length();
                while (total_read < content_length) {
                    concurrency::streams::container_buffer<std::string> isbuf;  // in-stream buffer
                    size_t byte_read = body.read(isbuf, 16*1024).get();
                    total_read += byte_read;
                    const std::string &data = isbuf.collection();
                    parser.AcceptSomeData((const char*)data.c_str(), (long)byte_read);
                }

                // Validate parameters
                std::map<std::string, MPFD::Field*> fields = parser.GetFieldsMap();
                std::map<std::string, MPFD::Field*>::iterator it;
                for (it=fields.begin(); it!=fields.end(); it++) {
                    if ((it->first == "image") && (fields[it->first]->GetType() == MPFD::Field::FileType)) {
                        img = fields[it->first]->GetFileContent();
                        img_size = fields[it->first]->GetFileContentSize();
                        img_filename = fields[it->first]->GetFileName();
                    }
                    else if ((it->first == "threshold") && (fields[it->first]->GetType() == MPFD::Field::TextType)) {
                        threshold = std::stod(fields[it->first]->GetTextTypeContent());
                        if ((threshold > 1) || (threshold < 0)) {
                            threshold = -1;
                        }
                    }
                    else if ((it->first == "abs") && (fields[it->first]->GetType() == MPFD::Field::TextType)) {
                        auto temp = fields[it->first]->GetTextTypeContent();
                        // convert content to lower case
                        std::for_each(temp.begin(), temp.end(), [](char& c) {
                            c = ::tolower(c);
                        });
                        if (temp == "true") {
                            abs = true;
                        }
                    }
                    else {
                        status = status_codes::BadRequest;
                        jsn["error"] = json::value::string("Invalid parameter");
                        std::cout << "Invalid parameter" << std::endl;
                        break;
                    }
                }

                if (status == status_codes::OK) {
                    if (img == NULL || img_size == 0) {
                        status = status_codes::BadRequest;
                        jsn["error"] = json::value::string("Cannot find image");
                        std::cout << "Cannot find image" << std::endl;
                    }
                    else {
                        std::cout << "Inference request (image size: " << img_size << "; threshold: " << threshold 
                                  << "; normalized: " << !abs << ")" << std::endl;

                        std::cout << "Infering..." << std::flush;
                        auto t1 = std::chrono::high_resolution_clock::now();
                        auto cvimg = ie->openImage(img, (size_t)img_size);
                        auto t2 = std::chrono::high_resolution_clock::now();
                        auto inference = ie->infer(cvimg);
                        auto t3 = std::chrono::high_resolution_clock::now();
                        jsn = ie->parse(inference, !abs, threshold);
                        auto t4 = std::chrono::high_resolution_clock::now();
                        status = status_codes::OK;

                        ms t_rx    = std::chrono::duration_cast<ms>(t1 - t0);
                        ms t_load  = std::chrono::duration_cast<ms>(t2 - t1);
                        ms t_infer = std::chrono::duration_cast<ms>(t3 - t2);
                        ms t_parse = std::chrono::duration_cast<ms>(t4 - t3);
                        ms t_total = std::chrono::duration_cast<ms>(t4 - t0);
                        std::cout << " done (rx: " << t_rx.count() << "mS; load: " << t_load.count() << "mS; infer: " << t_infer.count() 
                                  << "mS; parse: " << t_parse.count() << "mS; total: " << t_total.count() << "mS)" << std::endl;
                    }
                }
            }
            catch (MPFD::Exception ex) {
                status = status_codes::BadRequest;
                jsn["error"] = json::value::string(ex.GetError());
                std::cout << " " << ex.GetError() << std::endl;
            }
            catch (std::exception const &ex) {
                status = status_codes::InternalError;
                jsn["error"] = json::value::string(ex.what());
                std::cout << " " << ex.what() << std::endl;
            }
        }
        else {
            status = status_codes::BadRequest;
            jsn["error"] = json::value::string("Invalid header (cannot find content-type)");
            std::cout << "Invalid header (cannot find content-type)" << std::endl;
        }
    }
    request.reply(status, jsn);
}

void handle_put(http_request request) {
    auto t0 = std::chrono::high_resolution_clock::now();
    http::status_code status = status_codes::OK;
    json::value jsn;
    auto uri = request.relative_uri();
    auto path = uri.path();
    std::cout << "---------- PUT " << uri.to_string() << std::endl;

    auto paths = http::uri::split_path(http::uri::decode(path));
    if ((paths.size() != 1) || (paths[0] != "model" && paths[0] != "labelmap")) {
        status = status_codes::NotFound;
        std::ostringstream stream;
        stream << "Path not found (" << path << ")";
        jsn["error"] = json::value::string(stream.str());
        std::cout << stream.str() << std::endl;
    }
    else {
        char *labelmap = NULL, *model_xml = NULL, *model_bin = NULL;
        unsigned long xml_size = 0, bin_size = 0, labelmap_size = 0;

        http_headers headers = request.headers();
        concurrency::streams::istream body = request.body();
        auto parser = MPFD::Parser();
        try {
            parser.SetUploadedFilesStorage(MPFD::Parser::StoreUploadedFilesInMemory);
            parser.SetMaxCollectedDataLength(std::numeric_limits<long>::max());
            parser.SetContentType(headers.content_type());

            // ref: https://docs.microsoft.com/zh-tw/previous-versions/jj950083%28v%3dvs.140%29
            size_t total_read = 0;
            auto content_length = headers.content_length();
            while (total_read < content_length) {
                concurrency::streams::container_buffer<std::string> isbuf;  // in-stream buffer
                size_t byte_read = body.read(isbuf, 16*1024).get();
                total_read += byte_read;
                const std::string &data = isbuf.collection();
                parser.AcceptSomeData((const char*)data.c_str(), (long)byte_read);
            }

            std::map<std::string, MPFD::Field*> fields = parser.GetFieldsMap();
            std::map<std::string, MPFD::Field*>::iterator it;
            for (it=fields.begin(); it!=fields.end(); it++) {
                if (fields[it->first]->GetType() == MPFD::Field::FileType) {
                    if (it->first == "xml") {
                        model_xml = fields[it->first]->GetFileContent();
                        xml_size = fields[it->first]->GetFileContentSize();
                    }
                    else if (it->first == "bin") {
                        model_bin = fields[it->first]->GetFileContent();
                        bin_size = fields[it->first]->GetFileContentSize();
                    }
                    else if (it->first == "labelmap") {
                        labelmap = fields[it->first]->GetFileContent();
                        labelmap_size = fields[it->first]->GetFileContentSize();
                    }
                    else {
                        status = status_codes::BadRequest;
                        std::ostringstream stream;
                        stream << "Invalid parameter/type (" << it->first << ")";
                        jsn["error"] = json::value::string(stream.str());
                        std::cout << "Invalid parameter/type (" << it->first << ")" << std::endl;
                        break;
                    }
                }
                else { // MPFD::Field::TextType
                    status = status_codes::BadRequest;
                    std::ostringstream stream;
                    stream << "Invalid parameter/type (" << it->first << ")";
                    jsn["error"] = json::value::string(stream.str());
                    std::cout << "Invalid parameter/type (" << it->first << ")" << std::endl;
                    break;
                }
            }

            if (status == status_codes::OK) {
                if (paths[0] == "model") {
                    if (model_xml == NULL || xml_size == 0 || model_bin == NULL || bin_size == 0) {
                        status = status_codes::BadRequest;
                        jsn["error"] = json::value::string("Cannot find model");
                        std::cout << "Cannot find model" << std::endl;
                    }
                    else {
                        std::cout << "Load model request (xml: " << xml_size << "; bin: " << bin_size << ")" << std::endl;

                        auto t1 = std::chrono::high_resolution_clock::now();

                        // Since OpenVINO cannot load model by file pointer or buffer, so we have to save it as a file
                        std::cout << "Loading..." << std::flush;
                        std::ofstream fp;
                        std::string filename_xml = "model.xml";
                        std::string filename_bin = "model.bin";
                        fp.open(filename_xml, std::ofstream::binary);
                        fp.write(model_xml, xml_size);
                        fp.close();
                        fp.open(filename_bin, std::ofstream::binary);
                        fp.write(model_bin, bin_size);
                        fp.close();

                        // Load model
                        ie->loadModel(filename_xml, filename_bin);

                        remove(filename_xml.c_str());
                        remove(filename_bin.c_str());

                        auto t2 = std::chrono::high_resolution_clock::now();
                        ms t_rx    = std::chrono::duration_cast<ms>(t1 - t0);
                        ms t_load  = std::chrono::duration_cast<ms>(t2 - t1);
                        ms t_total = std::chrono::duration_cast<ms>(t2 - t0);
                        std::cout << " done (rx: " << t_rx.count() << "mS; load: " << t_load.count() 
                                  << "mS; total: " << t_total.count() << "mS)" << std::endl;

                        status = status_codes::OK;
                        jsn["xml"] = json::value::number(xml_size);
                        jsn["bin"] = json::value::number(bin_size);
                    }
                }
                else {  // paths[0] == "labelmap"
                    if (labelmap == NULL || labelmap_size == 0) {
                        status = status_codes::BadRequest;
                        jsn["error"] = json::value::string("Cannot find labelmap");
                        std::cout << "Cannot find labelmap" << std::endl;
                    }
                    else {
                        std::cout << "Load labelmap" << std::endl;
                        std::cout << "    labelmap_size: " << labelmap_size << std::endl;
                        status = status_codes::OK;
                        jsn["labelmap"] = json::value::number(labelmap_size);
                    }
                }
            }
        }
        catch (MPFD::Exception ex) {
            status = status_codes::BadRequest;
            jsn["error"] = json::value::string(ex.GetError());
            std::cout << " " << ex.GetError() << std::endl;
        }
        catch (std::exception const &ex) {
            status = status_codes::InternalError;
            jsn["error"] = json::value::string(ex.what());
            std::cout << " " << ex.what() << std::endl;
        }
    }
    request.reply(status, jsn);
}

//void handle_del(http_request request) {
//    http::status_code status;
//    json::value jsn;
//    auto uri = request.relative_uri();
//    auto path = uri.path();
//    std::cout << "---------- DELETE " << uri.to_string() << std::endl;
//
//    auto paths = http::uri::split_path(http::uri::decode(path));
//    if ((paths.size() != 1) || (paths[0] != "model" && paths[0] != "labelmap")) {
//        status = status_codes::NotFound;
//        std::ostringstream stream;
//        stream << "Path not found (" << path << ")";
//        jsn["error"] = json::value::string(stream.str());
//    }
//    else {
//        status = status_codes::OK;
//        jsn["message"] = json::value::string("Hello there, this is a DELETE response.");
//    }
//    request.reply(status, jsn);
//}