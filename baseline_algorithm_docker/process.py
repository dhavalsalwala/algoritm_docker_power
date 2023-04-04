import glob
from pathlib import Path
from typing import List, Dict
from torch import nn
import pandas as pd
import numpy as np
import torch

from model.model_utils import get_ensemble_predictions, ensemble_uncertainties_regression
from model.myModel import ProbMCdropoutDNN
from model.tranformations import data_normalization, denorm_prediction
from utils.eval_utils import ShippingBaseAlgorithm, DEFAULT_MODEL_PATH, load_json, DEFAULT_INPUT_PATH, \
    DEFAULT_OUTPUT_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyAlgorithm(ShippingBaseAlgorithm):
    def __init__(self,
                 input_path: Path = DEFAULT_INPUT_PATH,
                 output_path: Path = DEFAULT_OUTPUT_PATH,
                 model_path: Path = DEFAULT_MODEL_PATH):
        super().__init__(input_path=input_path, output_path=output_path, model_path=model_path)
        # self.std = None
        # self.mean = None
        self.list_of_models: Dict[str,List[nn.Module]] = {}
        self.mean = {}
        self.std = {}
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
        # load member of ensembles
        baselines = glob.glob(str(DEFAULT_MODEL_PATH / Path("models/baselines_EOE_P*")))
        for baseline in baselines:
            # load scaling parameters
            self.mean[baseline.split("_")[6]] = pd.Series(load_json(Path(baseline) / Path("stats") / Path("means.json")))
            self.std[baseline.split("_")[6]] = pd.Series(load_json(Path(baseline) / Path("stats") / Path("stds.json")))

            members = glob.glob(str(Path(baseline) / Path("member_*")))
            self.list_of_models[baseline.split("_")[6]] = []
            for member_path in members:
                model = ProbMCdropoutDNN(input_size=len(self.input_features),
                                        hidden_size_1=50,
                                        hidden_size_2=20,
                                        dropout=0.005)
                model_member_path = Path(baseline) / Path(member_path) / Path("best_model.pth")
                model.load_state_dict(torch.load(model_member_path))
                model.to(device)
                self.list_of_models[baseline.split("_")[6]].append(model)

    @staticmethod
    def form_output(time_id, power, uncertainties):
        return pd.DataFrame({"time_id": time_id,
                             "power": power,
                             "uncertainty": uncertainties})

    def predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """
        Your algorithm goes here
        """

        do_part1 = inputs[(inputs["awind_vcomp_provider"]<=0)]
        do_part2 = inputs[(inputs["awind_vcomp_provider"]>0) & (inputs["awind_vcomp_provider"]<=10)]
        do_part3 = inputs[(inputs["awind_vcomp_provider"]>10) & (inputs["awind_vcomp_provider"]<=20)]
        do_part4 = inputs[(inputs["awind_vcomp_provider"]>20) & (inputs["awind_vcomp_provider"]<=30)]
        do_part5 = inputs[(inputs["awind_vcomp_provider"]>30)]

        preds_denorm = None
        indexes = []
        for partition in ["P1","P2","P3","P4","P5"]:

            if "P1" == partition:
                inputs = do_part1
            elif "P2" == partition:
                inputs = do_part2
            elif "P3" == partition:
                inputs = do_part3
            elif "P4" == partition:
                inputs = do_part4
            elif "P5" == partition:
                inputs = do_part5

            list_of_models = self.list_of_models[partition]
            mean = self.mean[partition]
            std = self.std[partition]

            # step 1. - data normalization
            norm_inputs = data_normalization(data=inputs[self.input_features],
                                            means=mean.loc[self.input_features],
                                            stds=std.loc[self.input_features])

            # step 2. - model predictions for each member of the ensembles
            tensor_inputs = torch.tensor(norm_inputs[self.input_features].values).float().to(device)

            preds_norm = get_ensemble_predictions(model_list=list_of_models,
                                                data_norm=tensor_inputs,
                                                multi_runs=10)

            # step 3. - denormalize predictions
            preds_denorm_local = denorm_prediction(preds_norm=preds_norm,
                                            target_mean=mean[self.target],
                                            target_std=std[self.target])
            if preds_denorm is None:
                preds_denorm = preds_denorm_local
            else:
                preds_denorm = np.concatenate((preds_denorm, preds_denorm_local), axis=1)
            
            indexes.extend(inputs.index)

        # step 4. calculate predictions & uncertainties of the ensemble

        # choose your uncertainty estimate
        uncertainties = ensemble_uncertainties_regression(preds=preds_denorm)['tvar']

        # avg the predictions of the ensembles
        avg_predictions = np.squeeze(np.mean(preds_denorm[:, :, 0], axis=0))

        # step 5. form the final dataframe!
        outputs = self.form_output(time_id=indexes,
                                   power=avg_predictions,
                                   uncertainties=uncertainties)
        return outputs


if __name__ == "__main__":
    MyAlgorithm().process()
