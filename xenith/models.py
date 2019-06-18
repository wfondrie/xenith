"""
Define models that Xenith can use.
"""
import os
import copy
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from xenith import torchmods
from xenith import dataset

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
    def __init__(self, model, source, num_features, pretrained=False,
                 feat_mean=None, feat_stdev=None, hidden_dims=None):
        """Instantiate an empty container"""
        self.model = model
        self.source = source
        self.num_features = num_features
        self.pretrained = pretrained
        self.feat_mean = feat_mean
        self.feat_stdev = feat_stdev
        self.hidden_dims = hidden_dims

    def save(self, file_name: str) -> None:
        """
        Save a XenithModel object to a file.

        Parameters
        ----------
        file_name : str
            The path to save the xenith model for future use.
        """
        model_spec = {"model_class": type(self.model).__name__,
                      "source": self.source,
                      "pretrained": self.pretrained,
                      "feat_mean": self.feat_mean,
                      "feat_stdev": self.feat_stdev,
                      "state_dict": self.model.state_dict(),
                      "num_features": self.num_features,
                      "hidden_dims": self.hidden_dims}

        torch.save(model_spec, file_name)

    def predict(self, psm_files: Tuple[str], gpu: bool = False) \
        -> "pandas.DataFrame":
        """
        Use a trained XenithModel to evaluate a new dataset.

        Parameters
        ----------
        psm_files : tuple of str or str
            The file path(s) to a collection of PSMs to evaluate with
            the model. These should not have been used for model
            training!

        gpu : bool
            Should the gpu be used, if available?

        Returns
        -------
        xenith.PsmDataset
             A PsmDataset object with the .
        """
        if self.pretrained == False:
            logging.warning("This model appears to be untrained!")

        if self.source == "percolator":
            normalize = False
        else:
            normalize = True

        device = _set_device(gpu)
        self.model.eval()

        psms = dataset.PsmDataset(psm_files, device,
                                  feat_mean=self.feat_mean,
                                  feat_stdev=self.feat_stdev,
                                  normalize=normalize)

        pred = self.model(psms.features)
        pred = pred.detach().cpu().numpy().flatten()
        psms.add_metric("xenith_score", pred)

        return(psms)


    def fit(self, training_files: Tuple[str],  validation_files: Tuple[str],
            max_epochs: int = 100, batch_size: int = 1028,
            learn_rate: float = 0.001, weight_decay: float = 0.01,
            early_stop: int = 5, gpu: bool = False, seed: int = 1) \
            -> "pandas.DataFrame":
        """
        Fit a XenithModel on a collection of cross-linked PSMs.

        The model is trained using the Adam algorithm to perform
        mini-batch gradient descent. 

        Parameters
        ----------
        training_files : tuple of str
            A training set of PSMs in the xenith tab-delimited format.

        validation_files : tuple of str
            A validation set of PSMs in xenith tab-delimited format.

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

        early_stop : int
            Stop training if the validation set loss does not decrease
            for `early_stop` consecutive epochs. Set to `None` to
            disable early stopping.

        gpu : bool
            Should the gpu be used, if available?

        seed : int
            The seed for random number generation.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the training and validation losses
            at each epoch.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = _set_device(gpu)

        train_set = dataset.PsmDataset(training_files, device)
        self.feat_mean = train_set.feat_mean
        self.feat_stdev = train_set.feat_stdev

        val_set = dataset.PsmDataset(validation_files, device,
                                     feat_mean=train_set.feat_mean,
                                     feat_stdev=train_set.feat_stdev)

        sig_loss = torchmods.SigmoidLoss()
        self.model = self.model.to(device)

        # Setup
        loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                             shuffle=True, drop_last=True)
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

            with torch.no_grad():
                self.model.eval()
                train_pred = self.model(train_set.features)
                train_loss = sig_loss(train_pred.flatten(), train_set.target)
                train_loss = train_loss.item()
                train_loss_tracker.append(train_loss)

                val_pred = self.model(val_set.features)
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

            _train_message(epoch, train_loss, val_loss)

            if np.isnan(loss):
                raise RuntimeError("NaN detected in loss.")

            if stop_counter == early_stop:
                logging.info("Stopping at epoch %s...", epoch)
                break

        res_msg = (f"Best Epoch = {best_epoch}, "
                   f"Validation Loss = {best_loss:.5f}")
        logging.info(res_msg)

        # Wrap-up
        self.model = best_model.cpu()
        self.pretrained = True
        self.source = "xenith"
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
        raise FileNotFoundError(f"{weights_file} not found.")

    weight_df = pd.read_csv(weights_file, sep="\t", nrows=2)
    weights = torch.FloatTensor(weight_df.loc[1, :].values)
    weights = weights[None, :] # Add a second dim

    bias = weights[:, -1]
    weights = weights[:, :-1]

    model = torchmods.Linear(input_dim=len(weights))
    model.linear.weight.data = weights
    model.linear.bias.data = bias

    return XenithModel(model=model, num_features=len(weights),
                       source="percolator", pretrained=True)


def load_model(xenith_model_file: str) -> "XenithModel":
    """
    Load a pretrained model from xenith.

    Parameters
    ----------
    xenith_model_file : str
        The saved model file output from xenith.

    Returns
    -------
    xenith.XenithModel
        A XenithModel object for predicting on a new dataset.
    """
    xenith_model_file = os.path.abspath(os.path.expanduser(xenith_model_file))
    if not os.path.isfile(xenith_model_file):
        raise FileNotFoundError(f"{xenith_model_file} not found.")

    model_spec = torch.load(xenith_model_file)

    if model_spec["model_class"] == "MLP":
        model = torchmods.MLP(input_dim=model_spec["num_features"],
                              layers=model_spec["hidden_dims"])
    else:
        model = torchmods.Linear(input_dim=model_spec["num_features"])

    model.load_state_dict(model_spec["state_dict"])

    return XenithModel(model=model, num_features=model_spec["num_features"],
                       hidden_dims=model_spec["hidden_dims"],
                       source=model_spec["source"],
                       pretrained=model_spec["pretrained"])


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

    return XenithModel(model=model, num_features=num_features,
                       hidden_dims=hidden_dims, source="xenith",
                       pretrained=False)


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
