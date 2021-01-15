import json, shlex, socket, subprocess, sys
import shlex,subprocess,os,io
# from google.cloud import speech
# from google.cloud.speech import enums
# from google.cloud.speech import types
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from .psphinx import transcribe_sphinx
import parameters as p
import numpy as np

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = p.GOOGLE_APPLICATION_CREDENTIALS

class RequestError(Exception): pass
class UnknownValueError(Exception): pass

class GSpeech(object):
    """Speech Recogniser using Google Speech API"""

    def __init__(self):
        """Constructor""" 
        self.online = is_connected()
        self.transcript = None
        self.started = True

    def do_recognition(self):
        show_all = False
        """Do speech recognition"""
        if not self.online:
            return self.offline_recognition()
            
        key = p.GOOGLE_API_KEY
        url = "http://www.google.com/speech-api/v2/recognize?{}".format(urlencode({
            "client": "chromium",
            "lang": "en-US",
            "key": key,
        }))
    
        
        while self.started:
            recording_path_stereo = str(p.TEMP_FOLDER+"recording_2.flac") 
            recording_path_mono = str(p.TEMP_FOLDER+"recording.flac") 
            sox_stereo_record = "sox -r 16000 -c 2 -t alsa "+p.DEVICE_INDEX+" "+recording_path_stereo+" silence 1 0.1 1% 1 1.3 1% trim 0 "+str(p.COMMAND_DURATION)
            sox_convert_to_mono = "sox "+recording_path_stereo+" -c 1 "+recording_path_mono
                       
            sox_p = subprocess.call(shlex.split(sox_stereo_record))
            sox_convert = subprocess.call(shlex.split(sox_convert_to_mono))
                                          
            with io.open(recording_path_mono,'rb') as audio_file:
                content = audio_file.read()
                request = Request(url, data=content, headers={"Content-Type": "audio/x-flac; rate={}".format(16000)})
            
            try:
                response = urlopen(request, timeout=None)                
            except Exception as e: 
                print(e)
                continue
            response_text = response.read().decode("utf-8")
            # ignore any blank blocks
            actual_result = []
            for line in response_text.split("\n"):
                if not line: continue
                result = json.loads(line)["result"]
                if len(result) != 0:
                    actual_result = result[0]
                    break

            # return results
            if show_all: return actual_result
            if not isinstance(actual_result, dict) or len(actual_result.get("alternative", [])) == 0: 
                print("Sorry could not recognize your command. Try again")
                continue

            if "confidence" in actual_result["alternative"]:
                # return alternative with highest confidence score
                best_hypothesis = max(actual_result["alternative"], key=lambda alternative: alternative["confidence"])
            else:
                # when there is no confidence available, we arbitrarily choose the first hypothesis.
                best_hypothesis = actual_result["alternative"]
            if "transcript" not in best_hypothesis[0]: 
                print("Sorry could not recognize your command. Try again")
                continue
            self.transcript = [best_hypothesis[i]["transcript"] for i in range(len(best_hypothesis))]
            return self.transcript

    def offline_recognition(self):        
        print("No Internet connection available")
        recording_path_stereo = str(p.TEMP_FOLDER+"recording_2.flac") 
        recording_path_mono = str(p.TEMP_FOLDER+"recording.raw") 
        sox_stereo_record = "sox -r 16000 -c 2 -t alsa "+p.DEVICE_INDEX+" "+recording_path_stereo+" silence 1 0.1 1% 1 1.3 1% trim 0 "+str(p.COMMAND_DURATION)
        sox_convert_to_mono = "sox "+recording_path_stereo+" -c 1 "+recording_path_mono
                       
        sox_p = subprocess.call(shlex.split(sox_stereo_record))
        sox_convert = subprocess.call(shlex.split(sox_convert_to_mono))
                                          
        with io.open(recording_path_mono, 'rb') as audio_file:
            content = audio_file.read()
        transcript = transcribe_sphinx(recording_path_mono)
        self.transcript = transcript
        print(transcript)
        return transcript

def is_connected():
    """Check if connected to Internet"""
    try:
        # check if DNS can resolve hostname
        remote_host = socket.gethostbyname("www.google.com")
        # check if host is reachable
        s = socket.create_connection(address=(remote_host, 80), timeout=5)
        return True
    except:
        pass
    return False

