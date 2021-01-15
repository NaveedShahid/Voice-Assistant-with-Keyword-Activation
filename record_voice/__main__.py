#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import time
import struct
import os
import soundfile
import argparse
import parameters as p
from .utils import buffer_to_wav, trim_silence
from rhasspysilence import WebRtcVadRecorder

from csv import writer
import sys
from sys import byteorder
from array import array
from struct import pack
from pathlib import Path

import wave

THRESHOLD = 600
CHUNK_SIZE = 512
RATE = 16000

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    snd_data = _trim(snd_data)
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    silence = [0] * int(seconds * RATE)
    r = array('h', silence)
    r.extend(snd_data)
    r.extend(silence)
    return r

# def record():

def record_templates(
    record_dir: Path,
    name_format: str,
    recorder: WebRtcVadRecorder,
    args: argparse.Namespace,
):
    """Record audio templates."""
    print("Reading 16-bit 16Khz mono audio from stdin...", file=sys.stderr)

    num_templates = 0

    try:
        print(
            f"Recording template {num_templates}. Please speak your wake word. Press CTRL+C to exit."
        )
        recorder.start()

        while True:
            # Read raw audio chunk
            chunk = sys.stdin.buffer.read(recorder.chunk_size)
            if not chunk:
                # Empty chunk
                break

            result = recorder.process_chunk(chunk)
            if result:
                audio_bytes = recorder.stop()
                audio_bytes = trim_silence(audio_bytes)

                template_path = record_dir / name_format.format(n=num_templates)
                template_path.parent.mkdir(parents=True, exist_ok=True)

                wav_bytes = buffer_to_wav(audio_bytes)
                template_path.write_bytes(wav_bytes)
             
                num_templates += 1
                if num_templates > 3:
                    break
                print(
                    f"Recording template {num_templates}. Please speak your wake word. Press CTRL+C to exit."
                )
                recorder.start()
    except KeyboardInterrupt:
        print("Done")

def record_multiple(name, keyword):
    print("Start speaking now....")
    for i in range(3):
        record_to_file(name,keyword, i)

def main():
    parser =  argparse.ArgumentParser(prog="recorder")

    parser.add_argument(
        "--name",
        help="Name of the speaker",
    )
    parser.add_argument(
        "--keyword",
        help="name of keyword for detection",
    )
    args = parser.parse_args()
    
    recorder = WebRtcVadRecorder(
        vad_mode=p.VAD_SENSITIVITY,
        silence_method=p.SILENCE_METHOD,
        current_energy_threshold=p.CURRENT_THRESHOLD,
        max_energy=p.MAX_ENERGY,
        max_current_ratio_threshold=p.MAX_CURRENT_RATIO_THRESHOLD,
        min_seconds=0.5,
        before_seconds=1,
    )
    
    keyword_dir= Path(os.path.join(p.USER_FOLDER,args.name,"keyword_dir/"))

    record_format = str(args.keyword+"-{n:02d}.wav")

    record_templates(keyword_dir, record_format, recorder, args)

    keyword_map = pd.read_csv(p.PROFILES_CSV)
    
    index = int(keyword_map.shape[0])
    

    new_profile = [index, 
                   args.name,
                   args.keyword,
                   keyword_dir,
                   p.PROBABILITY_THRESHOLD, 
                   p.MINIMUM_MATCHES,
                   p.AVERAGE_TEMPLATES,
                   p.SKIP_PROBABILITY_THRESHOLD]
    
    append_list_as_row(p.PROFILES_CSV, new_profile)
# -----------------------------------------------------------------------------

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mrode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
        
if __name__ == "__main__":
    main()

