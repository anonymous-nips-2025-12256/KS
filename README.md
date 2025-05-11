KS1 and KS2
======

How to use
------
### Dataset

* GloVe1M: https://ann-benchmarks.com/index.html#datasets
* Word, SIFT10M, GIST, Tiny5M and GloVe2M: https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html

### Compile

* Prerequisite : openmp, cmake
* Prepare and Compile:
    1. Go to the root directory of KS2 or KS1.
	2. Put the base set(.fvecs), query set(.fvecs) and groundtruth set(.ivecs) into XXX folder, where XXX is the name of dataset.
          4. For KS1, all_batches.bin denotes the generated projection vectors for KS1(S_pol).
	3. Check the parameter setting in the script run_XXX.sh.
    4. Execute the following commands:

```bash
$ cd /path/to/project
$ mkdir -p build && cd build
$ cmake .. && make -j
```

### Building HSNW+KS2 Index (only once)

```bash
$ bash run_XXX.sh
```

### Run HNSW+KS2 Index

```bash
$ bash run_XXX.sh
```

### Build and Run KS1 Index (only once)

```bash
$ bash run_XXX.sh
```

### Parameters
* Please refer to the scripts for the detailed parameter settings.
