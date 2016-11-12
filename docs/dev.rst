.. toctree::
   :maxdepth: 2

Developer Docs and Tips
-----------------------

Cheat Sheet Doing a Dev Release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Increment the build number in ``conda.recipe/dev/meta.yml``, then::

   cd ~/sources/wqio/conda.recipe
   conda build dev --channel=conda-forge --python=3.4
   conda build dev --channel=conda-forge --python=3.5
   conda convert ~/miniconda/conda-bld/linux-64/wqio-0.4.x-py3* -p all
   anaconda upload ~/miniconda/conda-bld/linux-64/*/wqio-0.4.x-py3*.tar.bz2 --label=dev

