# Tabular Transformers for Modeling Multivariate Time Series

This repository provides the pytorch source code, and data for tabular transformers (TabFormer). Details are described in the paper [Tabular Transformers for Modeling Multivariate Time Series](http://arxiv.org/abs/2011.01843), to be presented at ICASSP 2021.

### Requirements

- Python (3.7)
- Pytorch (1.6.0)
- HuggingFace / Transformer (3.2.0)
- scikit-learn (0.23.2)
- Pandas (1.1.2)

(X) represents the versions which code is tested on.

These can be installed using yaml by running :

```
conda env create -f setup.yml
```

---

### Tabular BERT

#### pretraining

To train a tabular BERT model on action history transaction dataset run :

```
$ python main.py --do_train --mlm --field_ce \
                 --field_hs 64 \
                 --output_dir [output_dir]
```

ex.)
```
$ python main.py --do_train --mlm --field_ce --field_hs 64 --output_dir ./output_pretraining/action_history/
```

#### fine-tuning

```
$ python tabformer_bert_fine_tuning.py
```


Description of some options (more can be found in _`args.py`_):

- `--mlm` for masked language model; option for transformer trainer for BERT
- `--field_hs` hidden size for field level transformer
- `--user_ids` option to pick only transacations from particular user ids.
