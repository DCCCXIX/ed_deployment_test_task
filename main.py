#!/usr/bin/env python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.FileHandler("app.log", "w", "utf-8")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-mn",
    "--model_name",
    nargs="?",
    default="bhadresh-savani/bert-base-uncased-emotion",
    # default="bhadresh-savani/bert-base-go-emotion",
    type=str,
    help="Name of a huggingface model to use",
    required=False,
)
args = vars(parser.parse_args())

import waitress
from flask import Flask, Response, request

import model_handler

app = Flask(__name__)
mh = model_handler.Model_Handler(model_name=args["model_name"])


@app.route("/emotion-detection", methods=["POST"])
def predict_post():
    try:
        text = request.get_json()["text"]
        logging.info(f"Request: {text}")
        result = mh.predict(text)
        logging.info(f"Result: {result}")
        return mh.predict(text)
    except Exception as e:
        logging.exception(f"Bad request")
        return Response("Bad request", status=400)


if __name__ == "__main__":
    waitress.serve(app, host="0.0.0.0", port=5000)
