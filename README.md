# EE250 Final Project: IoT Device for Smart Home Audio Detection

## Team Members
Leyaa George <leyaageo@usc.edu>

Rida Faraz <faraz@usc.edu>

## Setup Instructions

### Node 2: The Processor
First set up node 2, the processor, as its server needs to be running for us to send data to it.

Change directories into node 2's subfolder:
```cd node2-processor```

To install dependencies, run in command line: 
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
Setup the PostgreSQL Database:
```psql -U postgres -c "CREATE DATABASE sound_detection;```

Finally, start the server:
```python main.py```

### Node 1: The Recorder
Next, set up node 1 to allow us to capture audio data and send it to node 2.

Chnage directories into node 1's subfolder:
```cd node1-recoder```

Install requires dependencies by running in command line:
```pip install -r requirements.txt```

Create a .env file:
```
cat > .env << EOF
NODE2_SERVER_URL=http://localhost:8000
EOF
```

Start recording:
```python main.py```

## External Libraries
[Data Set](https://drive.google.com/drive/folders/1xA27s1DDCEcmnuguD2fvUZmhmx-bB55C?usp=sharing) (We created a custom data set to train our model. It can be accessed using the Google Drive folder)
Next.js (The Next.js framework helped us create our user interface.)

# Additional Information

## Description of Device
Clear description of what your IoT system is trying to achieve

## Block Diagram
Block diagram clearly shows relevant components and interactions

## Description of Technical Tools
2 Description of components, platforms, protocols used, and processing/visualization
techniques; open and transparent acknowledgement of use of any AI tools.

## Reflection on Development Process
discussion of limitations that demonstrates insights to their cause and
possible remediation, lessons learned

## Our X-Factor
Project originality and/or difficulty
