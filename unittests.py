import unittest

from transformers.utils.dummy_pt_objects import DataCollator

import main


class ControllerTestCase(unittest.TestCase):
    def setUp(self):
        self.app = main.app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client(self)
        self.payload = {"text": "a mock text for testing purposes"}
        self.text = "a mock phrase for testing purposes"

    def test_predict(self):
        result = main.mh.predict(self.text)

        assert type(result) is dict
        assert len(result) == main.mh.model.classifier.out_features
        assert not all(value == 0 for value in result.values())
        assert abs(sum(result.values()) - 1) < 0.000001

    def test_predict_post(self):
        response = self.client.post("/emotion-detection", json=self.payload)
        response_dict = response.json

        assert response.status_code == 200
        assert type(response_dict) is dict
        assert not all(value == 0 for value in response_dict.values())

    def test_predict_post_invalid(self):
        response = self.client.post("/emotion-detection", data=self.payload)

        assert response.status_code == 400


if __name__ == "__main__":
    unittest.main()
