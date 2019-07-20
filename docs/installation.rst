Installation
============

**xenith** is easy to install and is compatible with Python 3.6+

Dependencies
------------
**xenith** has the following dependencies:

+ Python 3.6+
+ NumPy_
+ Pandas_
+ PyTorch_

.. _NumPy: https://numpy.org/
.. _Pandas: https://pandas.pydata.org/
.. _PyTorch: https://pytorch.org/

Installing Python
-----------------
Prior to installing **xenith** Python 3.6+ must be installed. If you think you
already have a Python version that meets this requirement, you can check from
the command line with:

.. code-block:: console

   $ python3 --version
   Python 3.7.3 # My current Python version

If you need to install Python, we recommend the Miniconda Distribution. To do
so, navigate to the `Miniconda download page
<https://docs.conda.io/en/latest/miniconda.html>`_ and select the download for your
operating system. [#]_ From there, follow the `installation instructions
<https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ to
finish your Python installation.


Installing xenith
-----------------
**xenith** is available on the Python Package Index (PyPI) and is easily
installed with pip from the command line:

.. code-block:: console

  $ pip install xenith


Installing with pip will automatically install any of the missing Python package
dependencies.

.. note::
   Automatically install PyTorch_ will install the default version. This may not
   match your system configuration, particulary if you want to use a GPU. See
   the `PyTorch installation page <https://pytorch.org/get-started/locally/>`_
   for specific installation details. 

.. rubric:: Footnotes

.. [#] In the case of macOS, the '.pkg installer' will be easier option for most
       people.
