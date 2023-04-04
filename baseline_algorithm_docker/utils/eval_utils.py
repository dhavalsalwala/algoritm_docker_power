import json
from pathlib import Path
from typing import Dict

import pandas as pd

DEFAULT_OUTPUT_PATH = Path("/Users/dhaval/Projects/shifts/algoritm_docker_power/baseline_algorithm_docker/output")
DEFAULT_INPUT_PATH = Path("/Users/dhaval/Projects/shifts/algoritm_docker_power/baseline_algorithm_docker/test")
DEFAULT_MODEL_PATH = Path("/Users/dhaval/Projects/shifts/algoritm_docker_power/baseline_algorithm_docker/model")


def load_json(path: Path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


class ShippingBaseAlgorithm:
    def __init__(self,
                 input_path: Path = DEFAULT_INPUT_PATH,
                 output_path: Path = DEFAULT_OUTPUT_PATH,
                 model_path: Path = DEFAULT_MODEL_PATH):
        """
        Base class to inherit in order to implement algorithm docker.
        :param input_path: a Path to load inputs
        :param output_path: a Path to store outputs
        :param model_path: a Path to a model folder
        """
        self._input_path: Path = input_path
        self._output_path: Path = output_path
        self._model_path: Path = model_path
        pass

    def load_inputs(self) -> pd.DataFrame:
        """
        Read a pd.DataFrame from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        data = load_json(self._input_path / Path("merchant-vessel-features.json"))
        input_df = pd.DataFrame(data)
        input_df.set_index("time_id", inplace=True)
        return input_df

    @staticmethod
    def check_output(outputs: pd.DataFrame, inputs: pd.DataFrame):
        """
        validate outputs.
        :param outputs: pd.DataFrame with outputs
        :param inputs: pd.DataFrame with inputs
        :raises Error is any of the checks fails
        """
        assert not outputs.isna().mean().any()  # check for nan values
        assert outputs.shape[0] == inputs.shape[0]  # check number of predictions
        assert outputs.shape[1] == 3  # check number of columns
        assert not any(outputs.columns.intersection(['time_id' 'power' 'uncertainty']))  # check columns names

    def save(self, data_dict: Dict):
        """
        save a data to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        :param data_dict: a python dictionery with the predictions
        """
        _output_file = self._output_path / Path("vessel-power-estimates.json")
        with open(_output_file, "w") as f:
            f.write(json.dumps(data_dict))

    def process(self):
        """
        run the inference process.
        You must inherit the BaseClass and overwrite the self.load_model() and self.predict() template methods
        """
        input_df = self.load_inputs()
        self.load_model()
        outputs: pd.DataFrame = self.predict(inputs=input_df)
        self.check_output(outputs, inputs=input_df)
        self.save(outputs.to_dict(orient="records"))

    def load_model(self):
        """
        set self.model with a pytorch model from self._model_path attribute
        :return: None
        """
        raise NotImplementedError("You have implement o load model method to set a model in self.model")

    def predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """

        :param inputs: a pd.DataFrame with inputs.
        :return: A pd.DataFrame with outputs.
        """
        raise NotImplementedError("You have implement o predict to return a dataframe with the predictions")
