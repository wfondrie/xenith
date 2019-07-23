.. _percolator-vignette:

Use Percolator Models
=====================

Percolator_ is a semi-supervised learning algorithm for re-scoring traditional
shotgun proteomic datasets and was originally the recommended way to validate
results from Kojak_. Similarly to training a new model in **xenith**,
Percolator_ can run on a training dataset and the learned model can be used in
**xenith**.

To get started, you'll need a training dataset---A collection of PSMs to run
through Percolator_ that are separate from the dataset you'll eventually want to
evaluate. In this tutorial, we'll assume you've used Kojak_ as your search
engine and thus have files such as ``train_set.kojak.txt``,
``train_set.perc.intra.txt``, and ``train_set.perc.inter.txt``.

You'll also need to download and install Percolator_. In this tutorial, we'll
demonstrate running Percolator_ and using **xenith** from the command line with
a Percolator_ model. 

1. Convert Kojak results for Percolator
---------------------------------------------------

Percolator_ can accept a tab-delimited format that is similar to the **xenith**,
dubbed the Percolator INput format (PIN). The conversion utility for Kojak_
results in **xenith** also allows you to convert to PIN format as well, ensuring
that the features used to train a model in Percolator_ are the same as those
used in **xenith**.

To convert the training dataset to PIN format, we can run:

.. code-block:: console
   :caption: Command Line

   $ xenith kojak --output_file train_set.pin --to_pin True \
     train_set.kojak.txt train_set.perc.inter.txt \
     train_set.perc.intra.txt


.. _Kojak: http://kojak-ms.org
.. _Percolator: http://percolator.ms

2. Run Percolator
-----------------

Assuming Percolator_ is in your path, run Percolator_ on the new PIN file. The
``--weights`` option save the weights for the linear SVM model to a file which
**xenith** can use.

.. code-block:: console
   :caption: Command Line

   $ percolator --weights weights.txt train_set.pin


3. Evaluate a new dataset with xenith
-------------------------------------

We're now ready to evaluate a new dataset in **xenith** using the model learned
by Percolator_. First, we need to convert the new dataset to the **xenith**
tab-delimited format:

.. code-block:: console
   :caption: Command Line

   $ xenith kojak --output_file new_set.txt --to_pin False \
     new_set.kojak.txt new_set.perc.inter.txt \
     new_set.perc.intra.txt

Then finally, we can apply the model learned by Percolator_ to evaluate the new
dataset in **xenith**:

.. code-block:: console
   :caption: Command Line

   $ xenith predict --model weights.txt new_set.txt

This will write ``xenith.psms.txt`` and ``xenith.xlinks.txt`` to the working
directory. 

