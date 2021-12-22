import unittest

import main


class ControllerTestCase(unittest.TestCase):
    def setUp(self):
        self.app = main.app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client(self)
        self.payload = {"text": "Sample text for testing purposes."}

    def test_predict(self):
        result = main.mh.predict(self.payload["text"])

        self.assertIs(type(result), dict)
        self.assertEqual(len(result), main.mh.model.model.classifier.out_features)
        self.assertTrue(not all(value == 0 for value in result.values()))
        self.assertTrue(abs(sum(result.values()) - 1) < 0.000001)

    def test_predict_post(self):
        response = self.client.post("/emotion-detection", json=self.payload)
        response_dict = response.json

        self.assertEqual(response.status_code, 200)
        self.assertIs(type(response_dict), dict)
        self.assertTrue(not all(value == 0 for value in response_dict.values()))

    def test_predict_post_invalid(self):
        response = self.client.post("/emotion-detection", data=self.payload)

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
