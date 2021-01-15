#!/usr/bin/env python3
import argparse
import os
import json
import logging
import sys
import threading
import struct
import time
import typing
import socket
import rhasspynlu

import numpy as np
import pandas as pd
import parameters as p
import sounddevice as sd

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from halo import Halo
from csv import writer
from rhasspysilence import WebRtcVadRecorder
from rhasspysilence.const import SilenceMethod
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from . import Raven, Template
from .gspeech import *
from .voice_recog import recognize_command, train_model, verify

if is_connected():
    _ONLINE = True
else:
    _ONLINE = False
    
_LOGGER = logging.getLogger("voice_assisted_control")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = p.GOOGLE_APPLICATION_CREDENTIALS

_EXIT_NOW = False
KEYWORD = None
SPEAKER = None

# -----------------------------------------------------------------------------
"""Main entry point."""
def _parser():
    parser = argparse.ArgumentParser(prog="voice_assisted_control")

    parser.add_argument(
        "--record",
        nargs="+",
        help="Record example templates to a directory",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.22,
        help="Normalized dynamic time warping distance threshold for template matching (default: 0.22)",
    )
    parser.add_argument(
        "--refractory-seconds",
        type=float,
        default=2.0,
        help="Seconds before wake word can be activated again (default: 2)",
    )
    parser.add_argument(
        "--print-all-matches",
        action="store_true",
        help="Print JSON for all matching templates instead of just the first one",
    )
    parser.add_argument(
        "--window-shift-seconds",
        type=float,
        default=Raven.DEFAULT_SHIFT_SECONDS,
        help=f"Seconds to shift sliding time window on audio buffer (default: {Raven.DEFAULT_SHIFT_SECONDS})",
    )
    parser.add_argument(
        "--dtw-window-size",
        type=int,
        default=5,
        help="Size of band around slanted diagonal during dynamic time warping calculation (default: 5)",
    )
    parser.add_argument(
        "--vad-sensitivity",
        type=int,
        choices=[1, 2, 3],
        default=p.VAD_SENSITIVITY,
        help="Webrtcvad VAD sensitivity (1-3)",
    )
    parser.add_argument(
        "--current-threshold",
        type=float,
        help="Debiased energy threshold of current audio frame",
    )
    parser.add_argument(
        "--max-energy",
        type=float,
        help="Fixed maximum energy for ratio calculation (default: observed)",
    )
    parser.add_argument(
        "--max-current-ratio-threshold",
        type=float,
        help="Threshold of ratio between max energy and current audio frame",
    )
    parser.add_argument(
        "--silence-method",
        choices=[e.value for e in SilenceMethod],
        default=SilenceMethod.VAD_ONLY,
        help="Method for detecting silence",
    )
    parser.add_argument(
        "--exit-count",
        type=int,
        default=1,
        help="Exit after some number of detections (default: never)",
    )
    parser.add_argument(
        "--read-entire-input",
        action="store_true",
        help="Read entire audio input at start and exit after processing",
    )
    parser.add_argument(
        "--max-chunks-in-queue",
        type=int,
        default=1,
        help="Maximum number of audio chunks waiting for processing before being dropped",
    )
    parser.add_argument(
        "--failed-matches-to-refractory",
        type=int,
        help="Number of failed template matches before entering refractory period (default: disabled)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    return parser

@dataclass
class RavenInstance:
    """Running instance of Raven (one per keyword)."""

    thread: threading.Thread
    raven: Raven
    chunk_queue: "Queue[bytes]"


# -----------------------------------------------------------------------------

    
def main():
    keyword_map = pd.read_csv(p.PROFILES_CSV)
    speakers = keyword_map["name"]
    data_folders = keyword_map["keyword_file"]
    
    parser = _parser()
    parser.add_argument(
        "--name",
        default=speakers,
        help="Name of the speaker of WAV templates",
    )
    parser.add_argument(
        "--keyword",
        default=data_folders,
        help="Directory with WAV templates",
    )
    parser.add_argument(
        "--keyword_name",
        default=keyword_map["keywords"],
        help="Name of the WAV templates",
    )
    parser.add_argument(
        "--probability_threshold",
        default=keyword_map["probability_threshold"],
        help="Directory with WAV templates names",
    )
    parser.add_argument(
        "--minimum_matches",
        default=keyword_map["minimum_matches"],
        help="Directory with WAV templates names",
    )
    parser.add_argument(
        "--skip_probability_threshold",
        default=keyword_map["skip_probability_threshold"],
        help="Skip additional template calculations if probability is below this threshold",
    )
    parser.add_argument(
        "--average_templates",
        default=keyword_map["average_templates"],
        help="Average wakeword templates together to reduce number of calculations",
    )
    parser.add_argument(
        "--chunk-size",
        default=1920,
        help="Number of bytes to read at a time from standard in (default: 1920)",
    )
    args = parser.parse_args()
    
    #Enrollment
    while True:
        hotword_detection(args)
    
def command_detection(audio_path,text=''):
    global SPEAKER
    command_csv = pd.read_csv(os.path.join(p.USER_FOLDER,SPEAKER,"commands","commands.csv"))
    model_folder = os.path.join(p.USER_FOLDER,SPEAKER,"models/")
    keyword_dir = os.path.join(p.USER_FOLDER,SPEAKER,"keyword-dir/")
    commands_model_folder = os.path.join(p.USER_FOLDER,SPEAKER,"command_models/")
    
    commands = command_csv["command"]
    data_folders = command_csv["command_path"]

    for c, file in zip(commands, data_folders):
        train_model(c,file, commands_model_folder)
    
    if train_model and text=='':
        command_detected = recognize_command(SPEAKER, audio_path)
        return command_detected
    elif text is not None:
        verified = verify(SPEAKER, text, audio_path)
        if verified:
            return verified 
        else:
            return ''
    else:
        print("No WAV files found in the source directory.....")

            
def hotword_detection(args):
    global _EXIT_NOW
    global KEYWORD
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        
    print("Starting wakeword detection......")
    # Create silence detector.
    # This can be shared by Raven instances because it's not maintaining state.
    recorder = WebRtcVadRecorder(
        vad_mode=args.vad_sensitivity,
        silence_method=args.silence_method,
        current_energy_threshold=args.current_threshold,
        max_energy=args.max_energy,
        max_current_ratio_threshold=args.max_current_ratio_threshold,
        min_seconds=0.5,
        before_seconds=1,
    )
    if args.record:
        _LOGGER.info("Please use the record_voice.py for recording audio")
        sys.exit()

    assert not args.keyword.empty, "--keyword is required"

    # Instances of Raven that will run in separate threads
    ravens: typing.List[RavenInstance] = []

    # Queue for detections. Handled in separate thread.
    output_queue = Queue()

    # Load one or more keywords
    for i in range(len(args.keyword)):
        template_dir = Path(args.keyword[i])
        wav_paths = list(template_dir.glob("*.wav"))
        if not wav_paths:
            _LOGGER.warning("No WAV files found in %s", template_dir)
            continue

        # Load audio templates
        keyword_templates = [
            Raven.wav_to_template(p, name=str(p), shift_sec=args.window_shift_seconds)
            for p in wav_paths
        ]
        name = args.name[i]
        probability_threshold = args.probability_threshold[i]
        minimum_matches = args.minimum_matches[i]
        skip_probability_threshold = args.skip_probability_threshold[i]
        average_templates = args.average_templates[i]

        raven_args = {
            "templates": keyword_templates,
            "keyword_name": args.keyword_name[i],
            "name": name,
            "recorder": recorder,
            "probability_threshold": probability_threshold,
            "minimum_matches": minimum_matches,
            "distance_threshold": args.distance_threshold,
            "refractory_sec": args.refractory_seconds,
            "shift_sec": args.window_shift_seconds,
            "skip_probability_threshold": skip_probability_threshold,
            "failed_matches_to_refractory": args.failed_matches_to_refractory,
            "debug": args.debug,
        }

        # Apply settings
        
        if average_templates:
            _LOGGER.debug(
                "Averaging %s templates for %s", len(keyword_templates), template_dir
            )
            raven_args["templates"] = [Template.average_templates(keyword_templates)]

        # Create instance of Raven in a separate thread for keyword
        raven = Raven(**raven_args)
        chunk_queue: "Queue[bytes]" = Queue()

        ravens.append(
            RavenInstance(
                thread=threading.Thread(
                    target=detect_thread_proc,
                    args=(chunk_queue, raven, output_queue, args),
                    daemon=True,
                ),
                raven=raven,
                chunk_queue=chunk_queue,
            )
        )

    # Start all threads
    for raven_inst in ravens:
        raven_inst.thread.start()

    output_thread = threading.Thread(
        target=output_thread_proc, args=(output_queue,), daemon=True
    )

    output_thread.start()

    # -------------------------------------------------------------------------
    
    print("Waiting for wake word utterence.......", file=sys.stderr)
        
    if args.read_entire_input:
        audio_buffer = FakeStdin(sys.stdin.buffer.read())
    else:
#         audio_buffer = sys.stdin.buffer
        audio_buffer = sd.RawInputStream(
            samplerate=16000, blocksize=args.chunk_size,
            channels=1, dtype='int16',
            callback=None)
        
    audio_buffer.start()

    while True:
        # Read raw audio chunk
        chunk , _= audio_buffer.read(args.chunk_size)
        if not chunk or _EXIT_NOW:
            _EXIT_NOW = False
            if KEYWORD is not None:
                break

            # Add to all detector threads
        for raven_inst in ravens:
            raven_inst.chunk_queue.put(chunk)

    if not args.read_entire_input:
        # Exhaust queues
        _LOGGER.debug("Emptying audio queues...")
        for raven_inst in ravens:
            while not raven_inst.chunk_queue.empty():
                raven_inst.chunk_queue.get()

    for raven_inst in ravens:
        # Signal thread to quit
        raven_inst.chunk_queue.put(None)
        _LOGGER.debug("Waiting for %s thread...", raven_inst.raven.keyword_name)
        raven_inst.thread.join()

    # Stop recorder stream 
    audio_buffer.close()
    audio_buffer.abort()

    # Stop output thread
    output_queue.put(None)
    _LOGGER.debug("Waiting for output thread...")
    output_thread.join() 

    if KEYWORD is not None:
        DETECTED = False
        for i in range(p.ATTEMPTS):
            text = transcribe_recognize()
            print(text)
            if text == False:
                continue
            intents = detect_intent(text)
            if intents == False:
                print("Command not found. Try again")
            else:
                DETECTED = True
                intent_name = intents[0].intent.name
                intent_entity = intents[0].entities
                intent_text = intents[0].text
                print("###############################################")
                print("Intent of command:",intent_name)
                for en in intent_entity:
                    print("Entity in command:",en.entity,"=",en.value)
                print("Complete command:", intent_text)
                print("###############################################","\n")
                
                process_intent(intent_name, intent_entity, intent_text) #send intent to be processed by raspi
                break
        if not DETECTED:
            print("Command not found. Stopping command detection")

# -----------------------------------------------------------------------------
def transcribe_recognize():
    global SPEAKER
    try:
        command_csv = pd.read_csv(os.path.join(p.USER_FOLDER,SPEAKER,"commands","commands.csv"))
    except:
        print("commands.csv file not found for user:",SPEAKER)
        print("Please enroll some commands and try again.")
        print("Activating wakeword detection....")
        return False
    

    gspeech = GSpeech()
    if not gspeech.online:
        print("Please speak a command:")
        gspeech.offline_recognition()
        text = gspeech.transcript
        print(text)
        if text is not None and not text == []:
            for t in command_csv["command"]:
                score = fuzz.ratio(text, t)
                print(score)
                if score>p.OFFLINE_SENTENCE_MATCH_THRESHOLD:
                    return t   
            return ''   
    else:
        gspeech.do_recognition()
        texts = gspeech.transcript
        if texts is not None and not texts == []:
            for text in texts:
                for t in command_csv["command"]:
                    score = fuzz.ratio(text, t)
                    if score>p.ONLINE_SENTENCE_MATCH_THRESHOLD:
                        return t   
            return ''   
        else:
            return text
   
    
def detect_thread_proc(chunk_queue, raven, output_queue, args):
    """Template matching in a separate thread."""
    global _EXIT_NOW
    global KEYWORD
    global SPEAKER
    detect_tick = 0
    start_time = time.time()

    while True:

        if args.max_chunks_in_queue is not None:
            # Drop audio chunks to bring queue size back down
            dropped_chunks = 0
            while chunk_queue.qsize() > args.max_chunks_in_queue:
                chunk = chunk_queue.get()
                dropped_chunks += 1

            if dropped_chunks > 0:
                _LOGGER.debug("Dropped %s chunks of audio", dropped_chunks)

        chunk = chunk_queue.get()
        if chunk is None:
            # Empty chunk indicates we should exit
            break

        # Get matching audio templates (if any)
        matching_indexes = raven.process_chunk(chunk)
        if len(matching_indexes) >= raven.minimum_matches:
            detect_time = time.time()
            detect_tick += 1

            # Print results for matching templates
            for template_index in matching_indexes:
                template = raven.templates[template_index]
                distance = raven.last_distances[template_index]
                probability = raven.last_probabilities[template_index]

                output_queue.put(
                    {            
                        "speaker": raven.name,
                        "keyword": raven.keyword_name,
                        "template": template.name,
                        "detect_seconds": detect_time - start_time,
                        "detect_timestamp": detect_time,
                        "raven": {
                            "probability": probability,
                            "distance": distance,
                            "probability_threshold": raven.probability_threshold,
                            "match_seconds": raven.match_seconds,
                        },
                    }
                )
                
                KEYWORD = raven.keyword_name
                SPEAKER = raven.name

                if not args.print_all_matches:
                    # Only print first match
                    break

        # Check if we need to exit
        if (args.exit_count is not None) and (detect_tick >= args.exit_count):
            _EXIT_NOW = True

# -----------------------------------------------------------------------------

###########################################################################
                #send command to raspberry pi based on intent
###########################################################################
def process_intent(intent_name, intent_entity, intent_text):
    if intent_name == "Lights On":
        for en in intent_entity:
            if en.entity == "location":
                location = en.value
            elif en.entity == "appliance":
                appliance = en.value
        # Now tell raspi to switch on the appliance at location 
    elif intent_name == "SetLightColor":
        for en in intent_entity:
            if en.entity == "color":
                color = en.value
        # tell raspi to set the light color
        
    elif intent_name == "CallSomeone":
        for en in intent_entity:
            if en.entity == "contact":
                contact = en.value
        # tell raspi to call contact

    
    

# -----------------------------------------------------------------------------

def detect_intent(text):
    f = open(p.SENTENCES_TXT, "r")
    string = f.read()
    intents = rhasspynlu.parse_ini(string)
    graph = rhasspynlu.intents_to_graph(intents) 
    output = rhasspynlu.recognize(text, graph)
    if not output == []:
        return output
    else:
        return False

# -----------------------------------------------------------------------------


def output_thread_proc(dict_queue):
    """Outputs a line of JSON for each detection."""
    while True:
        output_dict = dict_queue.get()
        if output_dict is None:
            break

        print("###############################################","\n"+\
              "speaker:", output_dict['speaker'],"\n"+\
              "keyword:", output_dict['keyword'],"\n"+\
              "template:", output_dict['template'],"\n"+\
              "detect_seconds:", output_dict['detect_seconds'],"\n"+\
              "detect_timestamp:", output_dict["detect_timestamp"],"\n"+\
              "probability:",output_dict["raven"]["probability"],"\n"+\
              "###############################################")


        
# -----------------------------------------------------------------------------

class FakeStdin:
    """Wrapper for fixed audio buffer that returns empty chunks when exhausted."""

    def __init__(self, audio_bytes: bytes):
        self.audio_bytes = audio_bytes

    def read(self, n: int) -> bytes:
        """Read n bytes from buffer or return empty chunk."""
        if len(self.audio_bytes) >= n:
            chunk = self.audio_bytes[:n]
            self.audio_bytes = self.audio_bytes[n:]
            return chunk

        # Empty chunk
        return bytes()

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
