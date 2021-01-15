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

from csv import writer
import sys
from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave

THRESHOLD = 700
CHUNK_SIZE = 1024
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

def record():

#     pa = pyaudio.PyAudio()
#     stream = pa.open(format=FORMAT, channels=1, rate=RATE,
#         input=True, output=True,
#         frames_per_buffer=CHUNK_SIZE)
    stream = sys.stdin.buffer
    
    num_silent = 0
    snd_started = False

    r = array('h')
    start = time.time()
    print("Press CTRL+C to stop recording")
    try:
        while (time.time()-start)<3:
            # little endian, signed short
            snd_data = array('h', stream.read(CHUNK_SIZE))
            if byteorder == 'big':
                snd_data.byteswap()
            r.extend(snd_data)

    except KeyboardInterrupt:
        pass
    finally:
        sample_width = 2
        r = normalize(r)
        r = trim(r)
        r = add_silence(r, 0.5)
        return sample_width, r

def record_to_file(name, command, index):
    sample_width, data = record()

    data = pack('<' + ('h'*len(data)), *data)
    user_dir = os.path.join(p.USER_FOLDER,name)
    commands_dir = os.path.join(user_dir,"commands/")
    command_dir = os.path.join(commands_dir,command)
    if not os.path.exists(user_dir):
        print("User with the name",name,"is not registered. Exitting...")
    if not os.path.exists(commands_dir):
        os.mkdir(commands_dir)
    if not os.path.exists(command_dir):
        os.mkdir(command_dir)
    path = os.path.join(command_dir,command+str(index)+".wav")
    
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        print(sample_width)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(data)
        wf.close()


def record_multiple(name, keyword):
    print("Start speaking now")
    for i in range(3):
        record_to_file(name,keyword, i)

def main():
    parser =  argparse.ArgumentParser(prog="recorder")

    parser.add_argument(
        "--name",
        help="Name of the speaker",
    )
    parser.add_argument(
        "--command",
        help="Name of command",
    )
    args = parser.parse_args()
    
    commands_dir = os.path.join(p.USER_FOLDER,args.name,"commands/")
    command_csv_path = os.path.join(commands_dir,"commands.csv")
    if not os.path.exists(command_csv_path):
        df = pd.DataFrame(columns=["command","command_path"])
        df.to_csv(command_csv_path)
    
    record_multiple(args.name, args.command)

    command_csv = pd.read_csv(command_csv_path)
    command_dir = os.path.join(commands_dir,args.command)
    index = int(command_csv.shape[0])
    
    record_format = str(args.command+"-{n:02d}.wav")
    new_profile = [index, 
                   args.command,
                   command_dir]
    
    append_list_as_row(command_csv_path, new_profile)
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

