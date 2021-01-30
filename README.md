# python-demo

Hello, welcome to my python demo code.

These python scripts have been extracted from different projects. Usually these files cannot be run as standalone programs.

1. The 2 llvmIR scripts uses the llvmpy/llvmlight library: https://llvmlite.readthedocs.io/en/latest/. It was an experiment before deciding to use LLVM C++ API  directly -- for better support. (the python bindings are just a nice wrapper for LLVM C++ API)

2. The prototype demo -- some scripts extracting from my Master Thesis prototype. I excluded the whole C++ engine and what you can see there are just some python scripts for data modelling ( extracting raw data, feature extraction, etc.) and some ML training and prediction scripts ( using SkLearn libary).
I know it might be difficult to understand, because there is basically where " I've done my research and results" for the final paper. So, the files are big and not really organized, as in production.

3. A recipe recommendation script (based on some affinity calculation -- excluded from here) and another script which calls an external API for data page-by-page. 

