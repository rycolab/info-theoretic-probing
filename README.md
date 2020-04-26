# probing-mutual-info

This repository contains code accompanying the paper: [Information-Theoretic Probing for Linguistic Structure (Pimentel et al., ACL 2020)](https://arxiv.org/abs/2004.03061).
It is a study of probing using information theoretic concepts.


## Install Dependencies

Create a conda environment with
```bash
$ conda env create -f environment.yml
```
Then activate the environment and install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).
```bash
$ conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
$ # conda install pytorch torchvision cpuonly -c pytorch
$ pip install transformers
```
Download and install fasttext as described [in this link](https://fasttext.cc/docs/en/support.html#building-fasttext-python-module).

## Running the code

To run the code simply use the command
```bash
$ make LANGUAGE=<language-name> TASK=<task>
```
Where task can be either pos_tag or dep_label and language name can be any of: english, czech, basque, finnish, turkish, tamil, korean, marathi, urdu, telugu, indonesian.

This command will download UD data and fasttext embeddings, running the full random search exploration with 50 runs for the four word representations: bert, fast, onehot and random.
Results will be found in folder `checkpoints/<task>/<language-name>/<representation>/all_results.txt`


## Extra Information

#### Citation

If this code or the paper were usefull to you, consider citing it:


```bash
@inproceedings{pimentel-etal-2019-meaning,
    title = "Information-Theoretic Probing for Linguistic Structure",
    author = "Pimentel, Tiago and
    Valvoda, Josef and
    Maudslay, Rowan Hall and
    Zmigrod, Ran and
    Williams, Adina and
    Cotterell, Ryan",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2004.03061",
}
```


#### Contact

To ask questions or report problems, please open an [issue](https://github.com/rycolab/info-theoretic-probing/issues).

