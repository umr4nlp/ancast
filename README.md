# Ancast
AnCast metric for evaluating UMR semantic graphs.

## Install
`ancast` is currently available as a local, editable install only

```shell
$ pip install -e .
```

## Usage
Consider sample AMR/UMR files in `samples` dir

### CLI command

Currently, `output` file path is required and will contain analysis csv file.

```shell
$ ancast --help
usage: ancast++ Argparser [-h] -p PRED [PRED ...] [-pe PRED_EXT [PRED_EXT ...]] -g GOLD [GOLD ...] [-ge GOLD_EXT [GOLD_EXT ...]] [-o OUTPUT] [-s {snt,doc}] [-df {amr,umr}] [-c CNEIGHBOR] [-sc SENSE_COEFFICIENT] [--separate-1-and-2 [SEPARATE_1_AND_2]]
                          [--allow-reify [ALLOW_REIFY]] [--use-alignment [USE_ALIGNMENT]] [--use-smatch-top [USE_SMATCH_TOP]] [--weighted {snts,toks}] [--seed SEED] [--debug]

options:
  -h, --help            show this help message and exit
  -p PRED [PRED ...], --pred PRED [PRED ...]
                        (required) path to prediction file or dir (if dir, same filenames must exist in `gold`)
  -pe PRED_EXT [PRED_EXT ...], --pred-ext PRED_EXT [PRED_EXT ...]
                        if `args.pred` flag is a dir, which file extensions to consider
  -g GOLD [GOLD ...], --gold GOLD [GOLD ...]
                        (required) path to gold file or dir (if dir, same filenames must exist in `pred`)
  -ge GOLD_EXT [GOLD_EXT ...], --gold-ext GOLD_EXT [GOLD_EXT ...]
                        if `args.gold` flag is a dir, which file extensions to consider
  -o OUTPUT, --output OUTPUT
                        (optional) path to output analysis files(if new dir, should not contain comma in the base dirname)
  -s {snt,doc}, --scope {snt,doc}
                        whether to run ancast over AMR/UMR sentence-level graphs (`snt`) or to run ancast++ over UMR document-level graphs (`doc`)
  -df {amr,umr}, --data-format {amr,umr}
                        whether the input data are AMR(s) or UMR(s) (automatically set to `umr` if `--scope` is `doc`)
  -c CNEIGHBOR, --cneighbor CNEIGHBOR
                        coefficient for the importance of neighborhood information when broadcasting
  -sc SENSE_COEFFICIENT, --sense-coefficient SENSE_COEFFICIENT
                        importance of sense ID when comparing two concepts
  --separate-1-and-2 [SEPARATE_1_AND_2]
                        whether to combine one-hop and two-hop neighbors together
  --allow-reify [ALLOW_REIFY]
                        whether to apply reification before comparing graphs
  --use-alignment [USE_ALIGNMENT]
                        whether to use alignment information when establishing anchors
  --use-smatch-top [USE_SMATCH_TOP]
                        whether to add (TOP :root `ROOT_NODE`) edge
  --weighted {snts,toks}
                        whether to apply weighted average for ancast++ doc-level evaluation by (1) number of sentences or (2) number of tokens
  --seed SEED           random seed for reproducibility
  --debug               debug logging mode
```

### Python API with Files
`evaluate` accepts the same set of arguments described above.

```python
from ancast import evaluate

pred_fpath = "./samples/umr_test.txt"
gold_fpath = "./samples/umr_gold.txt"
out_fpath = "./samples/outputs"

out = evaluate(
    pred_fpath,
    gold_fpath, 
    out_fpath, 
    scope="doc"
)
```

### Python API with Strings
You can pass AMR/UMR files as strings and acquire fscores and detailed comparison directly through `evaluate`
```python
from ancast import evaluate

pred_strings = open("./samples/umr_test.txt", "r").read() # alternatively you can directly pass generated umrs/amrs packed in standard format
gold_strings = open("./samples/umr_gold.txt", "r").read()
# out_fpath = "./samples/outputs"

out = evaluate(
    pred_strings,
    gold_strings, 
    # out_fpath, 
    scope="doc",
    output_csv_as_string=True
)
```


## Cite

Please cite this paper for now:

```bibtex
@inproceedings{sun2024ancast,
    title = "Anchor and Broadcast: An Eﬀicient Concept Alignment Approach for Evaluation of Semantic Graphs",
    author = "Sun, Haibo and Nianwen Xue ",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation",
    month = may,
    year = "2024",
}
```
