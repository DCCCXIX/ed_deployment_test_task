## A simple solution for serving a bert-based emotion detection model
Test task: https://github.com/xbodx/ds-task-junior/blob/main/README.md

This implementation uses a pretrained emotion detection model for english language by Bhadresh Savani

https://huggingface.co/bhadresh-savani/bert-base-go-emotion

Served with flask and waitress

### Setup

 - Install pytorch following instructions in the link below:

https://pytorch.org/get-started/locally/

In order to use GPU, make sure that your pytorch and cuda versions are compatible.
Using GPU is highly recommended.

To set up prerequisites execute in bash:
```bash
pip install -r requirements.txt
```
macos users (additional step required):
```bash
brew install libomp
```
### Usage

To launch the script execute in bash:
```bash
python main.py
```
Script's functionality can be checked be sending a post request to localhost on 5000 port.
Example:
```
import requests
import json
r = requests.post("http://localhost:5000/emotion-detection", json={'text': "a mock text for testing purposes"})
content = r.content
proba_dict = json.loads(content.decode('utf-8'))
proba_dict
```
This code will return a dict of probabilities for a set of 28 emotions
```
import requests
r = requests.post("http://localhost:5000/emotion-detection", data={'text': "a mock text for testing purposes"})
r.content
```
This code will return a 400, "Bad request" in case request contents are not json
