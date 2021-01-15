import os
from pocketsphinx import Pocketsphinx

def transcribe_sphinx(path, language="en-US", grammar=None, show_all=False):
        # import the PocketSphinx speech recognition module

    language_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pocketsphinx-data", language)
    acoustic_parameters_directory = os.path.join(language_directory, "acoustic-model")
    language_model_file = os.path.join(language_directory, "lm.txt")
    phoneme_dictionary_file = os.path.join(language_directory, "pronounciation-dictionary.dict")

    config = {
        'hmm': acoustic_parameters_directory,
        'lm': language_model_file,
        'dict': phoneme_dictionary_file
        }
    ps = Pocketsphinx(**config)
    ps.decode(
        audio_file=path,
        buffer_size=2048,
        no_search=False,
        full_utt=False
    )
    text = str(ps.hypothesis())
    return ps.hypothesis()
