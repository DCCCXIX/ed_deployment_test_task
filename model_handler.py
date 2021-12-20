import logging

import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class BertForMultilabelSequenceClassification(BertForSequenceClassification):
    """
    Custom class for multilabel sequence classification
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.float().view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class Model_Handler:
    """
    Preloads the model from models repo and handles prediction calls
    """

    def __init__(self):
        self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
        logging.info(
            f"Running on {'CPU' if self.device.type == 'cpu' else torch.cuda.get_device_name(self.device.index)}"
        )
        self.model = BertForMultilabelSequenceClassification.from_pretrained("bhadresh-savani/bert-base-go-emotion").to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-go-emotion")

    def predict(self, text):
        try:
            input_ids = self.tokenizer.encode(text)
            input_ids_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)
            result = self.model(input_ids_tensor).logits[0]
            # converting logits to probabilities
            non_negative_vect = result + abs(result.min())
            proba = non_negative_vect * (1 / sum(non_negative_vect))
            result_dict = dict(zip(self.model.config.id2label.values(), [item.item() for item in proba]))
        except Exception as e:
            logging.exception("Failed to proccess input")
            # If failed to proccess an input - return an empty dict with zeroes as values
            result_dict = dict(zip(self.model.config.id2label.values(), [0.0 * self.model.config.num_labels]))
        return result_dict
