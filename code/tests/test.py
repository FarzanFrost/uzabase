import unittest
from src.run import CountWords
import yaml
from typing import Dict, Any


class TestAll(unittest.TestCase):
    def test_load_config(self,):
        """
        Tests if the configurations loaded are correct.
        """

        # Load configuration in CountWords class.
        config_path: str = './config/cfg.yaml'
        CountWords.load_config(config_path=config_path)

        # Loads configuration.
        config: Dict[Any, Any] = {}
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.assertEqual(CountWords._config, config)
    
    def test_preprocess_data(self,):
        text: str = """This is a test sentence, with commas, a single quote ('), a double quote ("), and a dash - right here."""
        preprocessed_text: str = CountWords.preprocess_data(text=text)

        self.assertNotIn('.', preprocessed_text)
        self.assertNotIn(',', preprocessed_text)
        self.assertNotIn('-', preprocessed_text)
        self.assertNotIn('"', preprocessed_text)
        self.assertNotIn("'", preprocessed_text)

if __name__ == "__main__":
    unittest.main()