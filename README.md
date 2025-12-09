# EE250 Final Project: IoT Device for Smart Home Audio Detection

## Team Members
* Leyaa George <leyaageo@usc.edu>

* Rida Faraz <faraz@usc.edu>

## Setup Instructions

### Node 2: The Processor
First, set up node 2, the processor, as its server needs to be running for us to send data to it.

1. Change directories into node 2's subfolder after cloning the repository:
```cd node2-processor```

2. Create virtual environment and install dependencies:
```
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. Finally, start the server:
```python main.py```

**Create a public link in another terminal if you want external devices to connect to your server (ie: node 1 to be an external device), or leave it as localhost if testing on the same device. For public link:**
```ngrok http 8000```

### Node 1: The Recorder
Next, set up node 1 to allow us to capture audio data and send it to node 2.

1. Change directories into node 1's subfolder:
```cd node1-recorder```

2. Create virtual environment and install dependencies:
```
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Create a .env file (Replace localhost with the public https ngrok link if connecting to external device):
```
touch .env
nano .env
```
Within the .env, paste ```NODE2_SERVER_URL=http://localhost:8000``` and then save and exit.

4. Start listening for sound events:
```python recorder.py```

### User Interface
View activity log by opening `http://localhost:8000` in node 2 device's browser. 

## External Libraries
* [Data Set](https://drive.google.com/drive/folders/1xA27s1DDCEcmnuguD2fvUZmhmx-bB55C?usp=sharing) (We created a custom data set to train our model. It can be accessed using the Google Drive folder.)
* `fastapi` (FastAPI to facilitate HTTP communication by defining our API endpoints.)
* `uvicorn` (ASGI server that runs our FastAPI app and listens on the specified port.)
* `librosa` (This library provided many features that allowed us to process the audio data, including MFCC, altering its sampling size, file extension, duration, etc.)
* `scikit-learn` (This library was used to implement the random classifier model that was trained using the custom data set.)
* `joblib` (This library was used to save the parameters of the random forest classifier so that it could be used by the processing node to generate inferences.)
* `sqlite` (This library is used to create the database that stores the audio classifcations, including the timestamp of detection, confidence level, and dB level.)
* `scipy` (This is used for WAV file saving and parsing)
* `ngrok` (This library was used to transform the localhost link hosting node 2 into a public link that could be accessed by any device, enabling node to node communication.)
* `sounddevice` (This records audio from the microphone.)
* Full Dependency List (From `requirements.txt`): `aiohappyeyeballs, annotated-types, anyio, async-timeout, attrs, audioread, beautifulsoup4, certifi, cffi, charset-normalizer, click, contourpy, cycler, decorator, docopt, exceptiongroup, fastapi, filelock, fonttools, frozenlist, h11, idna, importlib_resources, Jinja2, joblib, Js2Py, kiwisolver, lazy_loader, librosa, llvmlite, MarkupSafe, matplotlib, mpmath, msgpack, multidict, networkx, numba, numpy, packaging, pandas, pillow, pipwin, platformdirs, pooch, propcache, psycopg2-binary, PyAudio, pycparser, pydantic, python-dateutil, python-dotenv, python-multipart, pytz, requests, scikit-learn, scipy, scikit-learn, seaborn, six, sniffio, sounddevice, soundfile, soupsieve, soxr, starlette, sympy, threadpoolctl, torch, typing_extensions, tzdata, urllib3, uv, uvicorn, yarl, zipp`

## AI Logs
AI Logs are stored in the ai_logs directory. Rida's AI Log is titled rf_log.pdf, while Leyaa's AI Log is titled lg_log.pdf
