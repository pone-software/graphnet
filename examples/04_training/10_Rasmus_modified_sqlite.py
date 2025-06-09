###################################################################################################################################
# Notes to run this script with gpus 
""" 
Run line in terminal each session before running the training:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Do inside VENV to setup the packages needed:
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
"""

# If no gpus needed then you must change these lines
"""  
gpus = [3]  # Set to [0] for no gpus
"fit": {"gpus": gpus}, # change "gpus" to "0" if no gpus needed
"""
###################################################################################################################################

from typing import Dict, Callable, Any, List

import os
import torch
print("Torch Cuda Version: ", torch.version.cuda)
print("Torch Cuda Available: ", torch.cuda.is_available())
from torch_geometric.data import Data


# Model stuff
from graphnet.models.detector import Detector
from graphnet.models.graphs import KNNGraph, GraphDefinition
from graphnet.models.gnn import DynEdge
from graphnet.models import StandardModel
from graphnet.models.task.classification import BinaryClassificationTask
from graphnet.training.loss_functions import BinaryCrossEntropyLoss

# Data stuff
from graphnet.data.dataset import SQLiteDataset, EnsembleDataset
from graphnet.constants import LIQUIDO_GEOMETRY_TABLE_DIR
from graphnet.data.dataloader import DataLoader
from graphnet.training.labels import Label
from graphnet.data.utilities.sqlite_utilities import query_database

class PONEPrototype(Detector):
    """ Prototype Class for the PONE Geometry"""

    # Replace this with your table
    geometry_table_path = os.path.join(
        LIQUIDO_GEOMETRY_TABLE_DIR, "liquido_v1.parquet"
    )
    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "string"
    sensor_id_column = "dom_id"
    sensor_time_column = "dom_time"
    charge_column = "charge"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "dom_x": self._xyz,
            "dom_y": self._xyz,
            "dom_z": self._xyz,
            "dom_time": self._t,
            "charge": self._charge,
        }
        return feature_map

    def _xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 1e3

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1e6
    
    def _charge(self, x: torch.tensor) -> torch.tensor:
        return x/ 1e2
    
class StaticLabel(Label):
    """Class for producing a static label."""

    def __init__(
        self,
        key: str,
        value: float,
    ):

        self._value = torch.tensor([value])

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, graph: Data) -> torch.tensor:
        return self._value


def make_ensemble_dataset(signal_data_path: str, 
                            noise_data_path: str, 
                            signal_truth_table: str,
                            noise_truth_table: str,
                            truth: List[str],
                            signal_pulsemap: str,
                            noise_pulsemap: str,
                            features: List[str],
                            graph_definition: GraphDefinition,
                            target: str,
                            n_signal: int,
                            n_noise: int,
                            ) -> EnsembleDataset:

    signal_selection = query_database(database = signal_data_path,
                                      query = f'select event_no from {signal_truth_table}').sample(n_signal)['event_no'].tolist()
    noise_selection = query_database(database = noise_data_path,
                                      query = f'select event_no from {noise_truth_table}').sample(n_noise)['event_no'].tolist()
    
    signal_dataset = SQLiteDataset(path = signal_data_path,
                                   graph_definition=graph_definition,
                                   pulsemaps= [signal_pulsemap],
                                   features = features,
                                   truth = truth,
                                   truth_table=signal_truth_table,
                                   selection = signal_selection,
                                   labels = {target: StaticLabel(key = target, value = 1)},
                                   ) # Grabs all events
    
    noise_dataset = SQLiteDataset(path = noise_data_path,
                                   graph_definition=graph_definition,
                                   pulsemaps= [noise_pulsemap],
                                   features = features,
                                   truth = truth,
                                   truth_table=noise_truth_table,
                                   selection = noise_selection,
                                   labels = {target: StaticLabel(key = target, value = 0)},
                                    ) # Grabs all events
    return EnsembleDataset([signal_dataset, noise_dataset])
    



def main(args: Dict[str, Any]) -> None:

    
    # Define graph representation
    graph_definition = KNNGraph(detector=PONEPrototype(),
                                input_feature_names = args["features"])
    
    # Get DataLoader
    dataset = make_ensemble_dataset(signal_data_path=args["data_path_signal"],
                                    noise_data_path=args["data_path_noise"],
                                    signal_truth_table = args["signal_truth_table"],
                                    noise_truth_table = args["noise_truth_table"],
                                    truth = args["truth"],
                                    signal_pulsemap=args["pulsemap_signal"],
                                    noise_pulsemap=args["pulsemap_noise"],
                                    features = args["features"],
                                    graph_definition=graph_definition,
                                    target = args["target"],
                                    n_signal = args["n_signal"],
                                    n_noise = args["n_noise"])
    
    training_dataloader = DataLoader(dataset = dataset,
                                     batch_size=args["batch_size"],
                                     num_workers =  args["num_workers"],
                                     shuffle = True)

    # NOTE: This is not a real test dataloader!
    fake_test_dataloader = DataLoader(dataset = dataset,
                                     batch_size=args["batch_size"],
                                     num_workers =  args["num_workers"],
                                     shuffle = False)

    # Building model
    backbone = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean"],
    )

    task = BinaryClassificationTask(hidden_size=backbone.nb_outputs,
                                    target_labels=args["target"],
                                    loss_function=BinaryCrossEntropyLoss(),
                                    loss_weight = None)

    model = StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task],
        optimizer_kwargs={"lr": args["lr"]},
    )

    
    # Training model
    model.fit(
        training_dataloader,
        logger= None,
        **args["fit"]
    )

    # Get predictions
    additional_attributes = ["event_no", args["target"]]
    assert isinstance(additional_attributes, list)  # mypy


    results = model.predict_as_dataframe(
        fake_test_dataloader,
        additional_attributes=additional_attributes,
        gpus=args["fit"]["gpus"],
    )
    outdir = args["outdir"]
    print("Out Directory: ", outdir)
    # Save predictions and model to file
    os.makedirs(outdir, exist_ok=True)

    # Save results as .csv
    results.to_parquet(f"{outdir}/train_results.parquet")

    model.save(f"{outdir}/model.pth")

    # Save model config and state dict - Version safe save method.
    # This method of saving models is the safest way.
    model.save_state_dict(f"{outdir}/state_dict.pth")
    model.save_config(f"{outdir}/model_config.yml")


if __name__  == '__main__':
    basedir = "/mnt/research/IceCube/PONE/jp_pone_sim/k40sim/sqlite"
    gpus = [3]  # Set to [0] for no gpus
    lr = 1e-03
    batch_size = 10
    num_workers = 10
    n_noise = 20
    n_signal = 20
    outdir = '/mnt/gs21/scratch/robsonj3/modified_sqlite'  # must change to current user
    target = 'is_signal'
    features = ["dom_x", "dom_y", "dom_z", "dom_time", "charge"]
    truth = ['event_no']
    pulsemap_signal = 'PMTResponse_nonoise'
    pulsemap_noise = 'K40PulseMap'
    truth_table_noise = 'K40_truth'
    truth_table_signal = 'GenerateSingleMuons_39_pmtsim_pframe_truth'
    data_path_signal = f"{basedir}/signal/merged/merged.db"
    data_path_noise = f"{basedir}/noise/merged/merged.db"
    


    args ={'batch_size': batch_size,
           'pulsemap_signal': pulsemap_signal,
           "pulsemap_noise": pulsemap_noise,
           "noise_truth_table": truth_table_noise,
           "signal_truth_table": truth_table_signal,
           "features": features,
           "fit": {"gpus": gpus}, # change "gpus" to "0" if no gpus needed
           "lr": lr,
           "outdir": outdir,
           "features": features,
           "data_path_signal": data_path_signal,
           "data_path_noise": data_path_noise,
           "target": target,
           "truth": truth,
           "num_workers": num_workers,
           "n_noise": n_noise,
           "n_signal": n_signal,
           }
    main(args)