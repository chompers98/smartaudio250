# EE250 Final Project: IoT Device for Smart Home Audio Detection

## Team Members
Leyaa George <leyaageo@usc.edu>

Rida Faraz <faraz@usc.edu>

## Setup Instructions

### Node 2: The Processor
First, set up node 2, the processor, as its server needs to be running for us to send data to it.

Change directories into node 2's subfolder after cloning the repository:
```cd node2-processor```

To install dependencies, run installation in command line: 
```pip install -r requirements.txt```

Then create a .env file for settings configuration:
```
cat > .env << EOF
NODE2_SERVER_URL=http://localhost:8000
DB_HOST=localhost
DB_USER=postgres
DB_PASSWORD=password
DB_NAME=sound_detection
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_PHONE_FROM=+1234567890
NOTIFY_PHONE_TO=+1234567890
EOF
```

Set up the PostgreSQL Database:
```psql -U postgres -c "CREATE DATABASE sound_detection;```

Finally, start the server:
```python main.py```

Create a public link if you want external devices to connect to your server, or leave it as localhost if testing on the same device. For public link:
```ngrok http 8000```

### Node 1: The Recorder
Next, set up node 1 to allow us to capture audio data and send it to node 2.

Change directories into node 1's subfolder:
```cd node1-recoder```

Install required dependencies by running in command line:
```pip install -r requirements.txt```

Create a .env file (Replace localhost with the public ngrok link if connecting to external device):
```
cat > .env << EOF
NODE2_SERVER_URL=http://localhost:8000
EOF
```

Start listening for sound events:
```python recorder.py```

### User Interface
View activity log by opening http://localhost:8000 in node 2 device's browser. 

## External Libraries
* `[Data Set](https://drive.google.com/drive/folders/1xA27s1DDCEcmnuguD2fvUZmhmx-bB55C?usp=sharing)` (We created a custom data set to train our model. It can be accessed using the Google Drive folder.)
* `FastAPI` (FastAPI was used to constantly listen for data sent to node 2 and run the processing scripts. This allowed us to facilitate HTTP communication.)
* `librosa` (This library provided many features that allowed us to process the audio data, including altering its sampling size, file extension, duration, etc.)
* `scikit-learn` (This library was used to implement the random classifier model that was trained using the custom data set.)
* `joblib` (This library was used to save the parameters of the random forest classifier so that it could be used by the processing node to generate inferences.)
* `ngrok` (This library was used to transform the localhost link hosting node 2 into a public link that could be accessed by any device, enabling node to node communication.)
* Full Dependency List (From `requirements.txt`): aiohappyeyeballs, annotated-types, anyio, async-timeout, attrs, audioread, beautifulsoup4, certifi, cffi, charset-normalizer, click, contourpy, cycler, decorator, docopt, exceptiongroup, filelock, fonttools, frozenlist, h11, idna, importlib_resources, Jinja2, joblib, Js2Py, kiwisolver, lazy_loader, librosa, llvmlite, MarkupSafe, matplotlib, mpmath, msgpack, multidict, networkx, numba, numpy, packaging, pandas, pillow, pipwin, platformdirs, pooch, propcache, psycopg2-binary, PyAudio, pycparser, pydantic, python-dateutil, python-dotenv, pytz, requests, scikit-learn, scipy, seaborn, six, sniffio, soundfile, soupsieve, soxr, starlette, sympy, threadpoolctl, torch, twilio, typing_extensions, tzdata, urllib3, uv, uvicorn, yarl, zipp
