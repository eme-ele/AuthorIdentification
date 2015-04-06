.. TreeTagger Python Wrapper documentation master file, created by
   sphinx-quickstart on Tue Oct 13 19:53:29 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TreeTagger Python Wrapper's documentation!
=====================================================

Contents:

.. toctree::
   :maxdepth: 2

..
    Include documentation from the module itself.

.. automodule:: treetaggerwrapper
    :members:

.. autofunction:: PipeWriter
.. autofunction:: IsSGMLTag
.. autofunction:: SplitSGML
.. autofunction:: BlankToTag
.. autofunction:: maketransU
.. autofunction:: BlankToSpace
.. autofunction:: enable_debugging_log
.. autofunction:: main


Example of use
--------------

Here is an example of use of the treetaggerwrapper to tag a set
of files from a directory::

    import glob,os,codecs

    def tagfiles(fromdir,filter="*.txt",outext="tags",encoding="utf-8") :
        """Tag files from a directory.
        """
        for fname in glob.glob(os.path.join(fromdir,filter)) :
            f = codecs.open(fname,r,encoding=encoding)
            # To be continued.

..
    Removed from doc.

    Indices and tables
    ------------------

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`



