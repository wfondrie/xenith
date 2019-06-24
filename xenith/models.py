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

import xenith
from xenith import torchmods

class XenithModel():
    """
    A container class used to load pretrained models.

    Parameters
    ----------
    model : torch.nn.Module
        A pytorch model to use.

    source : str
        Was this trained in xenith, or loaded from Percolator?

    num_features : int
        The number of input features.

    pretrained : bool
        Is the model already trained?

    feat_mean : pandas.Series
    feat_stdev : pandas.Series
        Series containing the mean and standard deviations of features
        from the training set. Future datasets that this model is
        applied too must have features that match these.

    hidden_dims : tuple of int
        A list indicating the dimensions of hidden layers that are used
        in the model. `None` indicates a linear model.

    Attributes
    ----------
    model : torch.nn.Module
        A PyTorch model.

    source : str
        Indicates the origin of the model. In the case of 'percolator',
        normalization is disabled during prediction because the raw
        model weights are used.

    num_features : int
        The number of input features.

    pretrained : bool
        Is the model already trained?

    feat_mean : pandas.Series
    feat_stdev: pandas.Series
        Series containing the mean and standard deviations of features
        from the training set. Future datasets that this model is
        applied too must have features that match these.

    hidden_dims : tuple of int
        A list indicating the dimensions of hidden layers that are used
        in the model. `None` indicates a linear model.
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

    def count_parameters(self):
        """Return the number of trainable parameters in the model."""
        mod = self.model
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)

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

    def predict(self, xenith_dataset: xenith.dataset.XenithDataset,
                name: str = "XenithDataset", gpu: bool = False) \
        -> xenith.dataset.XenithDataset:
        """
        Use a trained XenithModel to evaluate a new dataset.

        Parameters
        ----------
        xenith_dataset : xenith.dataset.XenithDataset
            A XenithDataset object containing the PSMs to evaluate.
            These should not have been used for model training!

        name : str
            The name of the output column. This is added to the
            'predictions' attribute of the output XenithDataset. If this
            column already exists, the previous column will be
            overwritten.

        gpu : bool
            Should the gpu be used, if available?

        Returns
        -------
        xenith.dataset.XenithDataset
             A XenithDataset with the new scores written to the
            'predictions' attribute.
        """
        if not self.pretrained:
            logging.warning("This model appears to be untrained!")

        if self.source == "percolator":
            normalize = False
        else:
            normalize = True

        device = _set_device(gpu)
        self.model.eval().to(device)

        psms = xenith.dataset._PsmDataset(xenith_dataset,
                                          feat_mean=self.feat_mean,
                                          feat_stdev=self.feat_stdev,
                                          normalize=normalize)

        pred = self.model(psms.features.to(device))
        pred = pred.detach().cpu().numpy().flatten()
        out_dataset = copy.deepcopy(xenith_dataset)
        out_dataset.prediction[name] = pred

        return out_dataset

    def fit(self, training_set: xenith.dataset.XenithDataset,
            validation_set: xenith.dataset.XenithDataset,
            max_epochs: int = 100, batch_size: int = 128,
            learn_rate: float = 0.001, weight_decay: float = 0.001,
            early_stop: int = 5, gpu: bool = False) \
            -> pd.DataFrame:
        """
        Fit a XenithModel on a collection of cross-linked PSMs.

        The model is trained using the Adam algorithm to perform
        mini-batch gradient descent.

        Parameters
        ----------
        training_set : xenith.XenithDataset
            A training set of PSMs. These are the PSMs that the model
            learns from.

        validation_set : xenith.XenithDataset
            A validation set of PSMs. These PSMs are used to assess when
            the model is trained.

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

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the training and validation losses
            at each epoch.
        """
        device = _set_device(gpu)

        train_set = xenith.dataset._PsmDataset(training_set,
                                               feat_mean=None,
                                               feat_stdev=None,
                                               normalize=True)

        self.feat_mean = train_set.feat_mean
        self.feat_stdev = train_set.feat_stdev
        val_set = xenith.dataset._PsmDataset(validation_set,
                                             feat_mean=train_set.feat_mean,
                                             feat_stdev=train_set.feat_stdev,
                                             normalize=True)

        sig_loss = torchmods.SigmoidLoss()

        # Send everything to 'device'
        self.model = self.model.to(device)
        train_set.features.to(device)
        train_set.target.to(device)
        val_set.features.to(device)
        val_set.target.to(device)

        # Setup
        loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                             shuffle=True, drop_last=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate,
                                     weight_decay=weight_decay, amsgrad=True)

        # Set tracking variables
        best_epoch = 0
        best_loss = 0
        stop_counter = 0
        train_loss_tracker = []
        val_loss_tracker = []

        # The main training loop
        for epoch in range(max_epochs):
            # Evaluate and update trackers ------------------------------------
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

            # The important bit -----------------------------------------------
            loss = _train_batch(loader, self.model, optimizer, sig_loss)

            # Communication and error tracking --------------------------------
            _train_message(epoch, train_loss, val_loss)

            if np.isnan(loss):
                raise RuntimeError("NaN detected in loss.")

            if stop_counter == early_stop:
                logging.info("Stopping at epoch %s...", epoch)
                break

        res_msg = (f"Best Epoch = {best_epoch}, "
                   f"Validation Loss = {best_loss:.5f}")
        logging.info(res_msg)

        # Wrap-up -------------------------------------------------------------
        self.model = best_model.cpu()
        self.pretrained = True
        self.source = "xenith"
        loss_df = pd.DataFrame({"epoch": list(range(epoch+1)),
                                "train_loss": train_loss_tracker,
                                "val_loss": val_loss_tracker})

        return loss_df


# Functions -------------------------------------------------------------------
def from_percolator(weights_file: str) -> xenith.models.XenithModel:
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
    xenith.models.XenithModel
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

    # Dummy pd.Series to verify that the features are correct.
    dummy = pd.Series([0]*len(weights), index=weight_df.columns.tolist())

    return XenithModel(model=model, num_features=len(weights),
                       feat_mean=dummy, feat_stdev=dummy,
                       source="percolator", pretrained=True)


def load_model(xenith_model_file: str) -> xenith.models.XenithModel:
    """
    Load a pretrained model from xenith.

    Parameters
    ----------
    xenith_model_file : str
        The saved model file output from xenith.

    Returns
    -------
    xenith.models.XenithModel
        A XenithModel object for predicting on a new dataset.
    """
    xenith_model_file = os.path.abspath(os.path.expanduser(xenith_model_file))
    if not os.path.isfile(xenith_model_file):
        raise FileNotFoundError(f"{xenith_model_file} not found.")

    model_spec = torch.load(xenith_model_file)

    if model_spec["model_class"] == "MLP":
        model = torchmods.MLP(input_dim=model_spec["num_features"],
                              hidden_dims=model_spec["hidden_dims"])
    else:
        model = torchmods.Linear(input_dim=model_spec["num_features"])

    model.load_state_dict(model_spec["state_dict"])

    return XenithModel(model=model,
                       num_features=model_spec["num_features"],
                       hidden_dims=model_spec["hidden_dims"],
                       feat_mean=model_spec["feat_mean"],
                       feat_stdev=model_spec["feat_stdev"],
                       source=model_spec["source"],
                       pretrained=model_spec["pretrained"])


def new_model(num_features: int, hidden_dims: Tuple[int] = (8, 8, 8)) \
    -> xenith.models.XenithModel:
    """
    Create a new model.

    Parameters
    ----------
    num_features : int
        The number of features used as input for the model.

    hidden_dims : tuple of int
        A list indicating the dimensions of hidden layers to use in the model.
        If a linear model is wanted, set to `[]` or `None`.

    Returns
    -------
    xenith.models.XenithModel
        A XenithModel to be trained.
    """
    if not hidden_dims or hidden_dims is None:
        model = torchmods.Linear(input_dim=num_features)
    else:
        model = torchmods.MLP(input_dim=num_features, hidden_dims=hidden_dims)

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
