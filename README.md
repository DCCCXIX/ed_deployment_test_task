## A simple solution for serving a bert-based emotion detection model

This implementation uses a pretrained emotion detection model for english language by Bhadresh Savani
https://huggingface.co/bhadresh-savani/bert-base-go-emotion
Served with flask and waitress

### Setup

 - Install pytorch following instructions in the link below:

https://pytorch.org/get-started/locally/

In order to use GPU, make sure that your pytorch and cuda versions are compatible.
Using GPU is adviced.

To set up prerequisites do in bash:
```bash
pip install requirements.txt
```
macos users (additional step):
```bash
brew install libomp
```
