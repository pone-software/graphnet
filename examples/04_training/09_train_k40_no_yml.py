# Choice of graph representation, architecture, and physics task
from graphnet.models.detector.pone import PONE
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.task.classification import MulticlassClassificationTask
from torch_geometric.loader import DataLoader
from graphnet.data.dataset.parquet.parquet_dataset import ParquetDataset
from graphnet.data import GraphNeTDataModule
from torch.utils.data import random_split
from graphnet.data.dataset.dataset import EnsembleDataset

from graphnet.constants import EXAMPLE_OUTPUT_DIR, TEST_DATA_DIR


# Choice of loss function and Model class
from graphnet.training.loss_functions import MAELoss, CrossEntropyLoss
from graphnet.models import StandardModel

import torch
from torch_geometric.data import Data
from graphnet.training.labels import Label
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from graphnet.utilities.logging import Logger
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
import time
import wandb
import os
import pyarrow.parquet as pq

"""
Multiclassifer trained to determine if event is neutrino or K40 based on the energy of event.

"""

logger = Logger()

class SignalBackgroundLabel(Label):
    """Class for producing signal/background label based on PID."""
    def __init__(self):
        """Construct `SignalBackgroundLabel`."""
        super().__init__(key="signal_background_label")

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        if not hasattr(graph, "pid"):
            raise ValueError("The graph does not contain the 'pid' attribute required for labeling.")
        pid = graph.pid.item()  # Access the PID field
        label = 1 if pid in [14, -14] else 0  # 1 for signal, 0 for background
        return torch.tensor(label, dtype=torch.float32)

class WandbMetricsLogger(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch_time = time.time() - self.batch_start_time
        metrics = trainer.callback_metrics
        wandb.log({
            "batch_idx": batch_idx,
            "train_loss_batch": metrics.get("train_loss", None),
            "batch_time": batch_time,  # Log batch processing time
        })


graph_definition = KNNGraph(
    detector=PONE(),
    node_definition=NodesAsPulses(),
    nb_nearest_neighbours=8,
    input_feature_names=["dom_x", "dom_y", "dom_z", "dom_time", "charge"],  # Define features explicitly
)


signal = ParquetDataset(
    #path= f"{EXAMPLE_OUTPUT_DIR}/convert_i3_files/pone",
    path= "/mnt/research/IceCube/PONE/jp_pone_sim/k40sim/pone_script_test",
    pulsemaps="PMTResponse_nonoise",
    truth_table="GenerateSingleMuons_39_pmtsim_pframe_truth",
    features=["dom_x", "dom_y", "dom_z", "dom_time", "charge"],
    truth=["energy"],
    graph_definition = graph_definition,
)

background = ParquetDataset(
    #path= f"{EXAMPLE_OUTPUT_DIR}/convert_i3_files/pone",
    path="/mnt/research/IceCube/PONE/jp_pone_sim/k40sim/pone_script_test",
    pulsemaps="K40PulseMap",
    truth_table="K40PulseMap_truth",
    features=["dom_x", "dom_y", "dom_z", "dom_time", "charge"],
    truth=["energy"],
    graph_definition = graph_definition,
)

# MANY DEBUG STATEMENTS BELOW!

# Inspect the input Parquet files and subdirectories
def inspect_parquet_files(path):
    print(f"Inspecting Parquet files and subdirectories in: {path}")
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".parquet"):
                file_path = os.path.join(root, file)
                label = "Signal (Muons)" if "GenerateSingleMuons" in file_path or "PMTResponse_nonoise" in file_path else "Background (K40)"
                print(f"File: {file_path} ({label})")
                try:
                    table = pq.read_table(file_path)
                    print(f"  Columns: {table.column_names}")
                    print(f"  Number of rows: {table.num_rows}")
                    # Print the first few rows of each column
                    for column in table.column_names:
                        print(f"  First few rows of column '{column}': {table[column].to_pylist()[:5]}")
                except Exception as e:
                    print(f"  Failed to read file: {e}")

# inspect_parquet_files("/mnt/research/IceCube/PONE/jp_pone_sim/k40sim/pone_script_test")
# signal.add_label(SignalBackgroundLabel())
# background.add_label(SignalBackgroundLabel())

# graph_sig = signal[0]
# graph_bkg = background[0]
# graph_sig["signal_background_label"]
# graph_bkg["signal_background_label"]
# print("Signal graph: ", graph_sig)
# print("Background graph: ", graph_bkg)

# Read out the features and truth values in the signal and background datasets
print("Available truth table columns:", signal._truth_table)
print("Signal Dataset Features: ", signal._features)
print("Signal Dataset Truth: ", signal._truth)
print("Background Dataset Features: ", background._features)
print("Background Dataset Truth: ", background._truth)

#since background is way larger we want to subsample it
generator1 = torch.Generator().manual_seed(42)
subsampled_bkg, _ = random_split(background, [10, len(background) - 10], generator=generator1)
print("Subsampled_background: ", len(subsampled_bkg))

subsampled_signal, _ = random_split(signal, [10, len(signal) - 10], generator=generator1)
print("Signal_Subsampled: ", len(subsampled_signal))
# create the total dataset from now equally sized bkg and signal datasets
print("Creating EnsembleDataset...")
ensemble_dataset = EnsembleDataset([subsampled_signal, subsampled_bkg])  # change: subsampled_signal to signal
print(f"EnsembleDataset created. Length: {len(ensemble_dataset)}")

# and now we can do the split in train, val, test
dataset_length = len(ensemble_dataset)
print(f"Total dataset length: {dataset_length}")

train_size = int(0.8 * dataset_length)
val_size = int(0.1 * dataset_length)
test_size = dataset_length - train_size - val_size  # Ensure all samples are used
print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

print("Splitting dataset into train, validation, and test sets...")
train_set, val_set, test_set = random_split(
    ensemble_dataset, [train_size, val_size, test_size], generator=generator1
)
print(f"Train set length: {len(train_set)}, Validation set length: {len(val_set)}, Test set length: {len(test_set)}")

# Debugging DataLoader arguments
print("Creating DataLoader for train set...")
train_dataloader = DataLoader(train_set, batch_size=1, num_workers=0)
print(f"Train DataLoader created with batch_size=1 and num_workers=0. Length: {len(train_dataloader)}")

print("Creating DataLoader for validation set...")
validate_dataloader = DataLoader(val_set, batch_size=1, num_workers=0)
print(f"Validation DataLoader created with batch_size=1 and num_workers=0. Length: {len(validate_dataloader)}")

print("Creating DataLoader for test set...")
test_dataloader = DataLoader(test_set, batch_size=1, num_workers=0)
print(f"Test DataLoader created with batch_size=1 and num_workers=0. Length: {len(test_dataloader)}")

# Debugging iteration through train_dataloader
print("Iterating through train_dataloader...")
for i, batch in enumerate(train_dataloader):
    print("Value of the train dataloader loop: ", i)
    print(f"Processing batch {i + 1}/{len(train_dataloader)}...")
    print(f"Batch details: {batch}")
    # Add any specific processing logic here
print("Finished iterating through train_dataloader.")


# Represents the data as a point-cloud graph where each
# node represents a pulse of Cherenkov radiation
# edges drawn to the 8 nearest neighbours

backbone = DynEdge(
    nb_inputs=graph_definition.nb_outputs,
    global_pooling_schemes=["min", "max", "mean"],
)
task = MulticlassClassificationTask(
    hidden_size=backbone.nb_outputs,
    nb_outputs=2,
    target_labels="signal_background_label",
    loss_function=CrossEntropyLoss(
        options=[2, torch.int64]),
)

# Construct the Model with GPU settings passed via trainer_kwargs
model = StandardModel(
    graph_definition=graph_definition,
    backbone=backbone,
    tasks=[task],
)

# Initialize wandb
wandb_run = wandb.init(
    project="array-performance-1",
    config={
        "learning_rate": 0.02,
        "epochs": 1,
    },
)

# Add the WandbMetricsLogger callback
wandb_logger_callback = WandbMetricsLogger()

# This is where you break!
batch = next(iter(train_dataloader))
preds = model(batch)  # Check if this runs without errors
print(preds)

# Train the model with the callback
model.fit(
    train_dataloader,
    max_epochs=wandb_run.config["epochs"],
    callbacks=[wandb_logger_callback],  # Add the callback here
)

print("TRAIN MODEL HAS FINISHED")

# Predict and save results
results = model.predict_as_dataframe(
    dataloader=test_dataloader,
    additional_attributes=model.target_labels + ["event_no"],
)
# Save predictions and model to file
""" outdir = "/mnt/home/robsonj3/knn_output"
os.makedirs(outdir, exist_ok=True)
results.to_csv(f"{outdir}/results.csv")
model.save_state_dict(f"{outdir}/state_dict.pth") """
# model.save(f"{outdir}/model.pth")