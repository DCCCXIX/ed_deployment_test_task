#!/usr/bin/env python
import datetime
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.FileHandler("app.log", "w", "utf-8")
logger.addHandler(handler)
import os

import waitress
from flask import Flask, Response, request

import model_handler

app = Flask(__name__)
mh = model_handler.Model_Handler()


@app.route("/emotion-detection", methods=["POST"])
def predict_post():
    try:
        text = request.get_json()["text"]
        logging.info(f"[{datetime.datetime.now()}]Request: {text}")
        result = mh.predict(text)
        logging.info(f"[{datetime.datetime.now()}]Result: {result}")
        return mh.predict(text)
    except Exception as e:
        logging.exception(f"[{datetime.datetime.now()}]Bad request")
        return Response("Bad request", status=400)


if __name__ == "__main__":
    waitress.serve(app, host="0.0.0.0", port=5000)
