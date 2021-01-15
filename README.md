# Voice-Assistant-with-Keyword-Activation
A simple CLI python system that uses wakeword detection [Raven](https://github.com/rhasspy/rhasspy-wake-raven) and uses Google's STT and [rhasspy-nlu](https://github.com/rhasspy/rhasspy-nlu) for command recognition.

## Wake word detection
First, the user needs to enroll his voice using the following command

``$ cd voice_assisted_control``
``$ ./configure``
``$ make``
``$ make install``

For enrolling a user, type:

``$ bin/voice_assisted_control --record users/<your_name>/keyword_dir/ "Seeen_{n:02d}.wav"``
	
Note: Replace *<your_name>* with your user name and record atleast 3 files with clear audio (listen to make sure)
Next add the proper lines and parameters for the wake word detection in etc/keyword_map/keyword_map.csv
	1 Seeen	 Sunshine	users/Seen/keyword_dir/	0.45	1	1	0.2
	Note: wake word sensitivity is controlled by probability threshold, minimum matches depend on number of files
	(more than 1 increases detection time, average_templates increases robustness(1 means more processing time but 
	lower chance of false activation), skip_probability can be controlled to reduce process time, 0 means faster.

Now run the application

  ``$ cd voice_assisted_control ``
  ``$ bin/voice_assisted_control ``

The application will start listening for wake word. speak your enrolled word to activate, if the program does not catch wake word,
adjust the probability_threshold (lower means more sensitive) it in the keyword_map.csv

Once speech is detected, program will try connecting to Google Cloud api, if succesful, it will start listening for commands
It will listen and translate text once, if the command is not detected, the application will restart and listen for wake word

For command identification, some code needs to be altered to use an ini file, check out [rhasspy-nlu](https://github.com/rhasspy/rhasspy-nlu)

for this to work properly, you need to specify task based commands in a format like this in a sentence.ini file
    ``[LightOn]``
	``turn on [the] (living room lamp | kitchen light){name}``
You can add any commands to this file in a format as mentioned in the link. Once the command is recognized
a json format code is returned which can be used to trigger code from raspberry pi


For offline detection, only the registered commands are used as, for registering commands, use 
	``$ arecord -r 16000 -f S16_LE -c 1 -t raw | bin/record_command --name Naveed --command "call Anny"``

you can add multiple commands in this way.

for offline detection, gmm models are used over the tensorflow model I previously mentioned due to system constraints
(process time atleast 5 seconds on my workstation, will be more than 10 seconds on Rpi)
gmm model accuracy is low and depends on the quality of audio recorded and the number of samples trained
On the first run, application if not connected to WiFi will train the models and store them, next speak a command for recognition
Note that the commands formats in sentence.txt can be many for better command detection
like 
	``[LightOn]``
	``turn on [the] (living room lamp | kitchen light){name}``

which works for all
	__turn on living room lamp__
 	__turn on the living room lamp__
	__turn on kitchen light__
	__turn on the kitchen light__

The above mentioned recording scripts are for testing purposes only considering your main goal is to take audio files from android, 
store on rpi, that is why the csv file can be connected to android to control thresholds and keyword names. I do not know your use
case so I cannot help you with creating the sentence.ini file. Feel free to ask any question regarding the project, I will be available
if you want to understand any part of the code.

