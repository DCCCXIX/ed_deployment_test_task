import logging

import torch
from transformers import pipeline


class Model_Handler:
    """
    Preloads the model from models repo and handles prediction calls
    """

    def __init__(self, model_name="bhadresh-savani/bert-base-go-emotion"):
        self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
        logging.info(f"Running on {self.device}")
        logging.info(f"Using model: {model_name}")
        self.model = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True,
            device=-1 if self.device.type == "cpu" else self.device.index,
        )

    def predict(self, text):
        try:
            result = self.model(text)
            result = dict(zip([row["label"] for row in result[0]], [row["score"] for row in result[0]]))

            value_vect = torch.tensor(list(result.values()))
            non_negative_vect = value_vect + abs(value_vect.min())
            proba = non_negative_vect * (1 / sum(non_negative_vect))
            result_dict = dict(zip(self.model.model.config.id2label.values(), [item.item() for item in proba]))

        except Exception:
            logging.exception("Failed to proccess input")
            # If failed to proccess an input - return an empty dict with zeroes as values
            result_dict = dict(
                zip(self.model.model.config.id2label.values(), [0.0 * self.model.model.config.num_labels])
            )
        return result_dict
