/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdexcept>
#include <memory>
#include "reader_factory.h"
#include "file_source_reader.h"
#include "coco_file_source_reader.h"

std::shared_ptr<Reader> create_reader(ReaderConfig config) {
    switch(config.type()) {
        case StorageType ::FILE_SYSTEM:
        {
            auto ret = std::make_shared<FileSourceReader>();
            if(ret->initialize(config) != Reader::Status::OK)
                throw std::runtime_error("File reader cannot access the storage");
            return ret;
        }
        break;
        case StorageType ::COCO_FILE_SYSTEM:
        {
            auto ret = std::make_shared<COCOFileSourceReader>();
            if(ret->initialize(config) != Reader::Status::OK)
                throw std::runtime_error("COCO File reader cannot access the storage");
            return ret;
        }
        break;
        default:
            throw std::runtime_error ("Reader type is unsupported");
    }
}