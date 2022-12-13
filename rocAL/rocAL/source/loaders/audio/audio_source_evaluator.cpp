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

#include "audio_source_evaluator.h"
#include "audio_decoder_factory.h"
#include "reader_factory.h"
// void AudioSourceEvaluator::set_size_evaluation_policy(MaxSizeEvaluationPolicy arg)
// {
//     _samples_max.set_policy (arg);
//     _channels_max.set_policy (arg);
// }

size_t AudioSourceEvaluator::max_samples()
{
    return _samples_max.get_max();
}

size_t AudioSourceEvaluator::max_channels()
{
    return _channels_max.get_max();
}

AudioSourceEvaluatorStatus
AudioSourceEvaluator::create(ReaderConfig reader_cfg, DecoderConfig decoder_cfg)
{
    AudioSourceEvaluatorStatus status = AudioSourceEvaluatorStatus::OK;

    // Can initialize it to any decoder types if needed


    // _header_buff.resize(COMPRESSED_SIZE);
    _input_path = reader_cfg.path();
    // std::cerr<< "Input Path:: "<< _input_path;
    if(_input_path.back() != '/') {
        _input_path = _input_path + "/";
        // std::cerr<<"_inp_path"<<_input_path;
    }
    _decoder = create_audio_decoder(std::move(decoder_cfg));
    _reader = create_reader(std::move(reader_cfg));
    find_max_dimension();
    // std::cerr<< "\n Input Path:: "<< _input_path;

    return status;
}

void
AudioSourceEvaluator::find_max_dimension()
{
    // std::cerr<<"\n Comes here in find_max_dim";
    _reader->reset();
    // std::cerr<<"\n  _reader->count_items()"<<  _reader->count_items();

    while( _reader->count_items() )
    {
        // std::cerr<<"\n While Loop";
        size_t fsize = _reader->open();
        if( (fsize) == 0 )
            continue;
        // std::cerr<<" \n reader_id "<<_reader->file_path();
        // std::cerr<<" \n _input_path "<<_input_path;

        auto file_name = _reader->file_path();
        // std::cerr<< "\n File Name in audio_source_evaluator" << file_name;
        // _header_buff.resize(fsize);
        // auto actual_read_size = _reader->read_data(_header_buff.data(), fsize);
        // _reader->close();

        if(_decoder->initialize(file_name.c_str()) != AudioDecoder::Status::OK)
        {
            WRN("Could not initialize audio decoder for file : "+ _reader->id())
            continue;
        }
        int samples, channels;

        if(_decoder->decode_info(&samples, &channels) != AudioDecoder::Status::OK)
        {
            WRN("Could not decode the header of the: "+ _reader->id())
            continue;
        }
        if(samples <= 0 || channels <=0)
            continue;

        _samples_max.process_sample(samples);
        _channels_max.process_sample(channels);
        // std::exit(0);

    }
    // return the reader read pointer to the begining of the resource
    _reader->reset();
}

void
AudioSourceEvaluator::FindMaxSize::process_sample(unsigned val)
{
    // if(_policy == MaxSizeEvaluationPolicy::MAXIMUM_FOUND_SIZE)
    // {
        _max = (val > _max) ? val : _max;
    // }
    // if(_policy == MaxSizeEvaluationPolicy::MOST_FREQUENT_SIZE)
    // {
    //     auto it = _hist.find(val);
    //     size_t count = 1;
    //     if( it != _hist.end())
    //     {
    //         it->second =+ 1;
    //         count = it->second;
    //     } else {
    //         _hist.insert(std::make_pair(val, 1));
    //     }

    //     unsigned new_count = count;
    //     if(new_count > _max_count)
    //     {
    //         _max = val;
    //         _max_count = new_count;
    //     }
    // }
}