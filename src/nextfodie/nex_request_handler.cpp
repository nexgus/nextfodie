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
#include <exception>
#include <iostream>
#include <limits>
#include <map>
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
    }
    else {
        auto queries = http::uri::split_query(query);
        int possible_query_count = queries.size();
        if ((possible_query_count < 1) || (possible_query_count > 3)) {
            status = status_codes::BadRequest;
            jsn["error"] = json::value::string("Bad Request (invalid query count)");
        }
        else {
            if (queries.find("path") == queries.end()) {
                status = status_codes::BadRequest;
                jsn["error"] = json::value::string("Bad Request (cannot find path)");
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
                }
                else {
                    std::cout << "Inference request" << std::endl;
                    std::cout << "    image: " << imgpath << std::endl;
                    std::cout << "    threshold: " << threshold << std::endl;
                    std::cout << "    abs: " << abs << std::endl;

                    auto img = ie->openImage(imgpath);
                    auto inference = ie->infer(img);
                    jsn = ie->parse(inference, !abs, threshold);
                    status = status_codes::OK;
                }
            }
        }
    }
    request.reply(status, jsn);
}

void handle_post(http_request request) {
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
                        break;
                    }
                }

                if (status == status_codes::OK) {
                    if (img == NULL || img_size == 0) {
                        status = status_codes::BadRequest;
                        jsn["error"] = json::value::string("Cannot find image");
                    }
                    else {
                        std::cout << "Inference request" << std::endl;
                        std::cout << "    image size: " << img_size << std::endl;
                        std::cout << "    threshold:  " << threshold << std::endl;
                        std::cout << "    abs:        " << abs << std::endl;

                        auto cvimg = ie->openImage(img, (size_t)img_size);
                        auto inference = ie->infer(cvimg);
                        jsn = ie->parse(inference, !abs, threshold);
                        status = status_codes::OK;
                    }
                }
            }
            catch (MPFD::Exception ex) {
                status = status_codes::BadRequest;
                jsn["error"] = json::value::string(ex.GetError());
            }
            catch (std::exception const &ex) {
                status = status_codes::InternalError;
                jsn["error"] = json::value::string(ex.what());
            }
        }
        else {
            status = status_codes::BadRequest;
            jsn["error"] = json::value::string("Invalid header (cannot find content-type)");
        }
    }
    request.reply(status, jsn);
}

void handle_put(http_request request) {
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
                        break;
                    }
                }
                else { // MPFD::Field::TextType
                    status = status_codes::BadRequest;
                    std::ostringstream stream;
                    stream << "Invalid parameter/type (" << it->first << ")";
                    jsn["error"] = json::value::string(stream.str());
                    break;
                }
            }

            if (status == status_codes::OK) {
                if (paths[0] == "model") {
                    if (model_xml == NULL || xml_size == 0 || model_bin == NULL || bin_size == 0) {
                        status = status_codes::BadRequest;
                        jsn["error"] = json::value::string("Cannot find model");
                    }
                    else {
                        std::cout << "Load model" << std::endl;
                        std::cout << "    xml_size: " << xml_size << std::endl;
                        std::cout << "    bin_size: " << bin_size << std::endl;

                        // Since OpenVINO cannot load model by file pointer or buffer, so we have to save it as a file
                        std::cout << "Saving model files...";
                        std::ofstream fp;
                        std::string filename_xml = "model.xml";
                        std::string filename_bin = "model.bin";
                        fp.open(filename_xml, std::ofstream::binary);
                        fp.write(model_xml, xml_size);
                        fp.close();
                        fp.open(filename_bin, std::ofstream::binary);
                        fp.write(model_bin, bin_size);
                        fp.close();
                        std::cout << " done" << std::endl;

                        // Load model
                        std::cout << "Loading model...";
                        ie->loadModel(filename_xml, filename_bin);
                        std::cout << " done" << std::endl;

                        // Remove temp files
                        std::cout << "Erasing saved model files...";
                        remove(filename_xml.c_str());
                        remove(filename_bin.c_str());
                        std::cout << " done" << std::endl;

                        status = status_codes::OK;
                        jsn["xml"] = json::value::number(xml_size);
                        jsn["bin"] = json::value::number(bin_size);
                    }
                }
                else {  // paths[0] == "labelmap"
                    if (labelmap == NULL || labelmap_size == 0) {
                        status = status_codes::BadRequest;
                        jsn["error"] = json::value::string("Cannot find labelmap");
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
        }
        catch (std::exception const &ex) {
            status = status_codes::InternalError;
            jsn["error"] = json::value::string(ex.what());
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