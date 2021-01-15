from rhasspysilence.const import SilenceMethod

# Folder Paths
PROFILES_CSV="etc/keyword_map/keyword_map.csv"
TEMP_FOLDER = "tmp/"
USER_FOLDER = "users/"
SENTENCES_TXT = "sentences.txt"

# Command Detection
ATTEMPTS = 2 # Command detection attemps
COMMAND_DURATION = 4
DEVICE_INDEX = 'hw:2,0'
ONLINE_SENTENCE_MATCH_THRESHOLD = 90
OFFLINE_SENTENCE_MATCH_THRESHOLD = 75

# Wakeword Detection
MINIMUM_MATCHES = 1
PROBABILITY_THRESHOLD = 0.42
DISTANCE_THRESHOLD = 0.22
SKIP_PROBABILITY_THRESHOLD = 0
AVERAGE_TEMPLATES = 0
REFRACTORY_SECONDS = 2.0
CHUNK_SIZE = 1920

# Recorder(Voice Activity Detection) Settings
VAD_SENSITIVITY = 1 #1, 2, 3 (sensitive to least sensitive)
SILENCE_METHOD = SilenceMethod.VAD_ONLY
MAX_CURRENT_RATIO_THRESHOLD = None
MAX_ENERGY = None
CURRENT_THRESHOLD = None

# Google API Settings
GOOGLE_APPLICATION_CREDENTIALS = "<path/to/gcp credentials in json>"
GOOGLE_API_KEY = "<path/to/api/key>" #Test key, use your own if this expires
