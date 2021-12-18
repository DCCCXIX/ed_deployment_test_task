#!/usr/bin/env python
import os

import waitress
from flask import Flask, request

import model_handler

# , render_template, url_for, redirect

app = Flask(__name__)


@app.route("/emotion-detection", methods=["GET", "POST"])
def predict_form_post():
    text = request.args.get("text")
    return model_handler.mh.predict(text)


if __name__ == "__main__":
    waitress.serve(app, host="0.0.0.0", port=5000)
    # app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
