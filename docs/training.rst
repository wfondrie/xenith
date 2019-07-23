Train a New Model
=================

While using the pretrained models distributed with **xenith** is easy, there are
several reasons why you may want to train your own:

* **You want to use a search engine other than Kojak.** **xenith** is
  currently distributed with only models for Kojak_. However, any search engine
  that can save PSM features in the **xenith** tab-delimited format can be used
  to train new models.

* **The included models don't adequately represent your experiments.** The
  pretrained models distributed with **xenith** were trained on a diverse set of
  publicly available data from PRIDE_. Unfortunately, these are largely
  experiments performed using BS3 as a cross-linker and collected on Orbitrap
  mass spectrometers. You will likely find performance improves if you train a
  custom model for your experiments using data from prior experiments as a
  training set.

* **You want to use a custom feature set.** Using the pretrained models
  distributed with **xenith** requires that you use exactly the same features
  that were originally used to train the model. If you find that additional
  features are helpful, you can train new models that incorporate those
  features.

Fortunately, training a new model is also easy in **xenith**, but it requires
use of the Python interface and some knowledge of machine learning. If this
sounds daunting, see the :ref:`percolator-vignette` vignette for details on how to
train new models with Percolator and use them in **xenith**.

1. Load PSMs
------------

To train a new model, you'll need a training set and a validation set.

The training set is a collection of PSMs that should be independent from your
dataset of interest (to which you eventually want to apply your model). The
training set is used to optimize the model's parameters to best discriminate
between target and decoy PSMs.

Likewise, the validation set is also a collection of PSMs that should be
independent from your dataset of interest. The validation is used to prevent the
model from overfitting to your training set. For best results, the validation
set should be independent from the training set.

Once you have a training and validation set prepared in **xenith** tab-delimited
format (``train_set.txt`` and ``val_set.txt`` below), we first need to load the
PSMs into :meth:`xenith.dataset.XenithDataset` objects:

.. code-block:: python
   :caption: Python

   import xenith
   import pandas as pd

   train_set = xenith.load_psms("train_set.txt")
   val_set = xenith.load_psms("val_set.txt")

   

.. tip::
   The :meth:`xenith.load_psms` function can accept a list of PSM files in the
   **xenith** tab-delimited format. This makes it easy to aggregate PSMs from
   multiple experiments into one training or validation set.


2. Select a model
-----------------

**xenith** offers two types of models to choose from: linear models and
multilayer perceptrons (MLP) [#]_. Traditionally, linear models (such as the
linear SVM in Percolator) have been used to re-score PSMs in proteomics.
However, non-linear models---such as MLPs---are more flexible and have been
shown to be beneficial for re-scoring PSMs, though at the risk of overfitting.

The code below continues our example by creating both a linear and an MLP model.
These are stored as :meth:`xenith.models.XenithModel` objects.

.. code-block:: python
   :caption: Python

   # Create a linear model
   linear_model = xenith.new_model(num_features=24, hidden_dims=None)

   # Create a multilayer perceptron model
   mlp_model = xenith.new_model(num_features=24, hidden_dims=(8, 8, 8))


3. Train and save the model
---------------------------

:meth:`xenith.dataset.XenithModel` objects are trained using the
:meth:`xenith.dataset.XenithModel.fit()` method. This method offers a
variety of parameters to tune the model to your training set. Unfortunately,
these parameters will need to be explored for your dataset to find what works
best. For example, the greater the model complexity (more features, more hidden
layers, bigger layers, *etc.*), the more epochs will be needed to complete
training.

In the ideal case, model training will be complete when the loss from the
validation set ceases to decrease (or even increases again). The
:meth:`xenith.dataset.XenithModel.fit()` method returns a ``pandas.DataFrame``
that contains the loss for each epoch, which can be used to evaluate training
progress.

.. code-block:: python
   :caption: Python

   # Let's fit the multilayer perceptron model
   training_progress = mlp_model.fit(training_set=train_set,
                                     validation_set=val_set)


You'll also want to save the trained model:

.. code-block:: python
   :caption: Python

   mlp_model.save("mlp_model.pt")


.. tip::
   If you have a GPU and CUDA version that is compatible with PyTorch, you can
   set ``gpu=True`` to use the GPU for model training.


4. Apply the newly trained model
--------------------------------

The newly trained model can then be used exactly like the models included with
**xenith**. Suppose we now want to evaluate a new dataset, ``new_set.txt``,
that is unrelated to the training and validation sets used for model training:

.. code-block:: python
   :caption: Python

   # Load the new dataset
   new_set = xenith.load_psms("new_set.txt")

   # Calculate scores with our new model
   scores = mlp_model.predict(new_set)

   # Add the new scores to the dataset
   new_set.add_metric(values=scores, name="score")

   # Estimate q-values
   psms, xlinks = new_set.estimate_qvalues(metric="score")

   # Save PSMs and cross-links to files.
   psms.to_csv("xenith.psms.txt", sep="\t", index=False)
   xlinks.to_csv("xenith.xlinks.txt", sep="\t", index=False)


.. note::
   When publishing results obtained from **xenith** using a custom model, the
   model file should **always** be shared as well! This will allow others to
   reproduce your results.


.. _Kojak: http://kojak-ms.org
.. _PRIDE: https://www.ebi.ac.uk/pride/archive


.. rubric:: Footnotes

.. [#] An MLP model is a type of artificial neural network where each layer of
       neurons is fully-connected to the adjacent layers. The `scikit-learn
       documentation
       <https://scikit-learn.org/stable/modules/neural_networks_supervised.html>`_
       is a good resource to learn more about MLP models.
