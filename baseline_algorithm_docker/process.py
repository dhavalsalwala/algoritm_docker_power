import glob
from pathlib import Path
from typing import List
from torch import nn
import pandas as pd
import numpy as np
import torch

from model.model_utils import get_ensemble_predictions, ensemble_uncertainties_regression
from model.myModel import ProbMCdropoutDNN
from model.tranformations import data_normalization, denorm_prediction
from utils.eval_utils import ShippingBaseAlgorithm, DEFAULT_MODEL_PATH, load_json, DEFAULT_INPUT_PATH, \
    DEFAULT_OUTPUT_PATH


class MyAlgorithm(ShippingBaseAlgorithm):
    def __init__(self,
                 input_path: Path = DEFAULT_INPUT_PATH,
                 output_path: Path = DEFAULT_OUTPUT_PATH,
                 model_path: Path = DEFAULT_MODEL_PATH):
        super().__init__(input_path=input_path, output_path=output_path, model_path=model_path)
        self.std = None
        self.mean = None
        self.list_of_models: List[nn.Module] = []
        # the input features order as in the training phase
        self.input_features = ["draft_aft_telegram",
                               "draft_fore_telegram",
                               "stw",
                               "diff_speed_overground",
                               "awind_vcomp_provider",
                               "awind_ucomp_provider",
                               "rcurrent_vcomp",
                               "rcurrent_ucomp",
                               "comb_wind_swell_wave_height",
                               "timeSinceDryDock"]
        self.target = "power"

    def load_model(self):
        # load scaling parameters
        self.mean = pd.Series(load_json(DEFAULT_MODEL_PATH / Path("stats") / Path("means.json")))
        self.std = pd.Series(load_json(DEFAULT_MODEL_PATH / Path("stats") / Path("stds.json")))

        # load member of ensembles
        members = glob.glob(str(DEFAULT_MODEL_PATH / Path("models/member_*")))
        for member_path in members:
            model = ProbMCdropoutDNN(input_size=len(self.input_features),
                                     hidden_size_1=50,
                                     hidden_size_2=20,
                                     dropout=0.005)
            model_member_path = Path(member_path) / Path("best_model.pth")
            model.load_state_dict(torch.load(model_member_path))
            self.list_of_models.append(model)

    @staticmethod
    def form_output(time_id, power, uncertainties):
        return pd.DataFrame({"time_id": time_id,
                             "power": power,
                             "uncertainty": uncertainties})

    def predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """
        Your algorithm goes here
        """
        # step 1. - data normalization
        norm_inputs = data_normalization(data=inputs[self.input_features],
                                         means=self.mean.loc[self.input_features],
                                         stds=self.std.loc[self.input_features])

        # step 2. - model predictions for each member of the ensembles
        tensor_inputs = torch.tensor(norm_inputs[self.input_features].values).float()

        preds_norm = get_ensemble_predictions(model_list=self.list_of_models,
                                              data_norm=tensor_inputs,
                                              multi_runs=10)

        # step 3. - denormalize predictions
        preds_denorm = denorm_prediction(preds_norm=preds_norm,
                                         target_mean=self.mean[self.target],
                                         target_std=self.std[self.target])

        # step 4. calculate predictions & uncertainties of the ensemble

        # choose your uncertainty estimate
        uncertainties = ensemble_uncertainties_regression(preds=preds_denorm)['tvar']

        # avg the predictions of the ensembles
        avg_predictions = np.squeeze(np.mean(preds_denorm[:, :, 0], axis=0))

        # step 5. form the final dataframe!
        outputs = self.form_output(time_id=inputs.index,
                                   power=avg_predictions,
                                   uncertainties=uncertainties)
        return outputs


if __name__ == "__main__":
    MyAlgorithm().process()
