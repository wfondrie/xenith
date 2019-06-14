"""
Define models that Xenith can use.
"""
import os
import copy
import pickle
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from xenith import torchmods


class XenithModel():
    """
    A container class used to load pretrained models.

    Parameters
    ----------
    model : torch.nn.Module
        A pytorch model to use.

    datasets : list or str
        A list of datasets used to train a model. In the case of a
        Percolator model, this will be `None`.

    model_type : str
        Indicates the type of model, either "percolator" or
        "xenith_mlp".

    features : list of str
        The feature names used to train the model, in order.
    """
    def __init__(self, model, source, datasets=None, pretrained=False):
        """Instantiate an empty container"""
        self.model = model
        self.source = source
        self.datasets = datasets
        self.pretrained = pretrained

    def save(self, file_name: str) -> None:
        """
        Save a XenithModel object
        """
        pass

    def fit(self, psm_files: Tuple[str],  max_epochs: int = 100,
            batch_size: int = 1028, learn_rate: float = 0.001,
            weight_decay: float = 0.01, val_fraction: float = 0.3,
            early_stop: int = 5, gpu: bool = False, seed: int = 1) -> None:
        """
        Fit a XenithModel on a collection of cross-linked PSMs.

        The model is trained using the Adam algorithm to perform
        mini-batch gradient descent. 

        Parameters
        ----------
        psm_files : tuple of str
            A training set of PSMs in the xenith tab-delimited format.

        max_epochs : int
            The maximum number of iterations through the full dataset to
            perform. As a simple rule, more epochs are needed for
            larger models and more complicated datasets.

        batch_size : int
            The batch size to use for gradient descent.

        learn_rate : float
            The learning rate of the Adam optimizer.

        weight_decay : float
            Adds L2 regularization to all model parameters.

        val_fraction : float
            The proportion of the dataset to be held out for validation.
            This validation set is how early stopping criteria are
            assessed.

        early_stop : int
            Stop training if the validation set loss does not decrease
            for `early_stop` consecutive epochs. Set to `None` to
            disable early stopping.

        gpu : bool
            Should the gpu be used?

        seed : int
            The seed for random number generation.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = _set_device(gpu)

        train_set = torch.zeros(1)
        val_set = torch.zeros(1)
        sig_loss = torchmods.SigmoidLoss()

        # Steps:
        # 1. Load and combine PSM data. All must have the same features.
        # 2. Execute training loop
        # 3. Set attributes for test set.

        # Send things to the gpu...
        val_set = val_set.to(device)
        self.model = self.model.to(device)

        # Setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate,
                                     weight_decay=weight_decay)

        # Set tracking variables
        best_epoch = 0
        best_loss = 0
        stop_counter = 0
        train_loss_tracker = []
        val_loss_tracker = []

        # The main training loop
        for epoch in range(max_epochs):
            loss = _train_batch(loader, self.model, optimizer, sig_loss)
            train_loss_tracker.append(loss)

            with torch.no_grad():
                self.model.eval()
                val_pred = self.model(val_set.feat)
                val_loss = sig_loss(val_pred.flatten(), val_set.target)
                val_loss = val_loss.item()
                val_loss_tracker.append(val_loss)
                self.model.train()

            if val_loss < best_loss or not best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_model = copy.deepcopy(self.model)
                stop_counter = 0
            else:
                stop_counter += 1

            _train_message(epoch, loss, val_loss)

            if np.isnan(loss):
                logging.warning("Nan detected in loss.")
                break

            if stop_counter == early_stop:
                logging.info("Stopping early...")

        res_msg = (f"Best Epoch = {best_epoch}, "
                   f"Validation Loss = {best_loss:.5f}")
        logging.info(res_msg)

        # Wrap-up
        self.model = best_model.cpu()
        loss_df = pd.DataFrame({"epoch": list(range(epoch+1)),
                                "train_loss": train_loss_tracker,
                                "val_loss": val_loss_tracker})

        return loss_df


# Functions -------------------------------------------------------------------
def from_percolator(weights_file: str) -> "XenithModel":
    """
    Load a pretrained model from Percolator results.

    Parameters
    ----------
    weights_file : str
        The output weights from Percolator. This can be obtained using
        the '--weights' option when running Percolator on a training
        dataset.

    Returns
    -------
    xenith.XenithModel
        A XenithModel object for predicting on a new dataset using the
        Percolator weights.
    """
    weights_file = os.path.abspath(os.path.expanduser(weights_file))
    if not os.path.isfile(weights_file):
        raise FileNotFoundError("'weights_file' not found.")

    weight_df = pd.read_csv(weights_file, sep="\t", nrows=3)
    weights = torch.FloatTensor(weight_df[1].values)

    bias = weights[-1]
    weights = weights[:-1]

    model = torchmods.Linear(input_dim=len(weights))
    model.linear.weight.data = weights
    model.linear.bias.data = bias

    return XenithModel(model=model, source="percolator", pretrained=True)


def from_xenith(xenith_model_file: str) -> "XenithModel":
    """
    Load a pretrained model from xenith.

    Parameters
    ----------
    xenith_model_file : str
        The saved model file output from xenith.

    Returns
    -------
    xenith.XenithModel
        A XenithModel object for predicting on a new dataset using the
        model that was trained in xenith.
    """
    xenith_model_file = os.path.abspath(os.path.expanduser(xenith_model_file))
    if not os.path.isfile(xenith_model_file):
        raise FileNotFoundError("'xenith_model_file' not found.")

    # TODO: Load files.

def new_model(num_features: int, hidden_dims: Tuple[int] = (8, 8, 8)):
    """
    Create a new model.

    Parameters
    ----------
    num_features : int
        The number of features used as input for the model.

    hidden_dims : tuple of int
        A list indicating the dimensions of hidden layers to use in the model.
        If a linear model is wanted, set to `[]` or `None`.
    """
    if not hidden_dims or hidden_dims is None:
        model = torchmods.Linear(input_dim=num_features)
    else:
        model = torchmods.MLP(input_dim=num_features, layers=hidden_dims)

    return XenithModel(model=model, source="xenith", pretrained=False)


# Utility functions -----------------------------------------------------------
def _set_device(gpu):
    """Set PyTorch to use the gpu, if requested."""
    gpu_avail = torch.cuda.is_available()
    device = torch.device("cuda:0" if gpu and gpu_avail else "cpu")
    if gpu and not gpu_avail:
        logging.warning("No gpu was detected. Using cpu instead.")

    return device


def _train_message(epoch, train_loss, val_loss):
    """Print messages about training progress."""
    msg = (f"Epoch {epoch}: Train loss = {train_loss:.5f}, "
           f"Validation Loss = {val_loss:.5f}")

    logging.info(msg)


def _train_batch(loader, model, optimizer, loss_fun):
    """Train a batch and return the loss"""
    running_loss = 0
    total = 0
    for batch_idx, (target, feat) in enumerate(loader):
        pred = model(feat)
        optimizer.zero_grad()
        loss = loss_fun(pred.flatten(), target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total = batch_idx

    return running_loss / (total + 1)


