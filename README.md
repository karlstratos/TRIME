# Setup

```
conda create --name trime python=3.7.11
conda activate trime
pip install --upgrade pip
export TMPDIR=/data/local/tmp/  # Annoying pip install [Errno 28] No space left on device
pip install --editable .
pip install torch==1.9.1+cu111  -f https://download.pytorch.org/whl/cu111/torch_stable.html  --no-cache-dir
conda install -c pytorch faiss-gpu
conda deactivate
```

# Data

```
bash get_data.sh wikitext-2
```

To examine how the data is preprocessed, after downloading the data explicitly run
```
python preprocess.py --only-source --trainpref data-bin/wikitext-2/raw_data/wikitext-2/wiki.train.tokens --validpref data-bin/wikitext-2/raw_data/wikitext-2/wiki.valid.tokens --testpref data-bin/wikitext-2/raw_data/wikitext-2/wiki.test.tokens --destdir data-bin/wikitext-2 --workers 1
```

## Toy/Synthetic Data
```
python preprocess.py --only-source --trainpref data-bin/toy/raw_data/toy/toy.train.tokens --validpref data-bin/toy/raw_data/toy/toy.valid.tokens --testpref data-bin/toy/raw_data/toy/toy.test.tokens --destdir data-bin/toy --workers 1
python preprocess.py --only-source --trainpref data-bin/synth/raw_data/synth/synth.train.tokens --validpref data-bin/synth/raw_data/synth/synth.valid.tokens --testpref data-bin/synth/raw_data/synth/synth.test.tokens --destdir data-bin/synth --workers 1
python train.py --task language_modeling data-bin/toy --save-dir /data/local/TRIME/output/toy --arch transformer_lm_wiki103_150M --criterion trime_loss --optimizer adam --adam-betas "(0.9, 0.98)" --weight-decay 0.01 --clip-norm 0.0 --max-update 200000 --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 8000 --warmup-init-lr 1e-07 --max-tokens 40 --update-freq 1 --tokens-per-sample 10 --seed 1 --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --adaptive-input --tie-adaptive-weights --adaptive-input-cutoff 20,30 --adaptive-softmax-cutoff 20,30 --knn-keytype last_ffn_input --fp16 --ce-warmup-epoch -1 --required-batch-size-multiple 1 --adaptive-softmax-factor 1 --adaptive-input-factor 1 --device-id 7 --distributed-world-size 1 --distributed-no-spawn  --reset-optimizer --reset-dataloader  --max-epoch 2
python train.py --task language_modeling data-bin/synth --save-dir /data/local/TRIME/output/synth --arch transformer_lm_wiki103_150M --criterion trime_loss --optimizer adam --adam-betas "(0.9, 0.98)" --weight-decay 0.01 --clip-norm 0.0 --max-update 200000 --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 8000 --warmup-init-lr 1e-07 --max-tokens 8 --update-freq 1 --tokens-per-sample 4 --seed 1 --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --adaptive-input --tie-adaptive-weights --adaptive-input-cutoff 2,4 --adaptive-softmax-cutoff 2,4 --knn-keytype last_ffn_input --fp16 --ce-warmup-epoch -1 --required-batch-size-multiple 1 --adaptive-softmax-factor 1 --adaptive-input-factor 1 --device-id 7 --distributed-world-size 1 --distributed-no-spawn  --reset-optimizer --reset-dataloader  --max-epoch 5
```

# Train

From `train_scripts/wiki103-150M-trime.sh` except I changed "20000,60000" to "2000,6000" for the adaptive stuff and added `--device-id 7 --distributed-world-size 1 --distributed-no-spawn` to use 1 GPU; also `--ce-warmup-epoch` set to -1 to study trime_loss from the beginning
```
python train.py --task language_modeling data-bin/wikitext-2 --save-dir /data/local/TRIME/output/wiki2-150M-trime --arch transformer_lm_wiki103_150M --criterion trime_loss --optimizer adam --adam-betas "(0.9, 0.98)" --weight-decay 0.01 --clip-norm 0.0 --max-update 200000 --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 8000 --warmup-init-lr 1e-07 --max-tokens 9000 --update-freq 2 --tokens-per-sample 150 --seed 1 --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --adaptive-input --tie-adaptive-weights --adaptive-input-cutoff 2000,6000 --adaptive-softmax-cutoff 2000,6000 --knn-keytype last_ffn_input --fp16 --ce-warmup-epoch 3 --required-batch-size-multiple 1 --adaptive-softmax-factor 1 --adaptive-input-factor 1 --device-id 7 --distributed-world-size 1 --distributed-no-spawn  --reset-optimizer --reset-dataloader --ce-warmup-epoch -1  # Add --data_debug data_debug.pkl to save training batches for debugging
```

# Eval

```
python eval_lm-trime.py data-bin/wikitext-103 --path /data/local/TRIME/pretrained_models/wiki103-150M-trime/checkpoint_best.pt --sample-break-mode complete --max-tokens 3072 --context-window 2560 --softmax-batch 1024 --gen-subset valid --fp16 --max-sentences 1 --knn-keytype last_ffn_input --use-local --softmax-temp 1.17
```

This gives

```
2022-12-09 14:09:48 | INFO | fairseq_cli.eval_lm | Evaluated 105971 tokens in 202.1s (524.29 tokens/s)                                                                                                                                                                                                                                                                   2022-12-09 14:09:48 | INFO | fairseq_cli.eval_lm | Loss (base 2): 4.6053, Perplexity: 24.34
Evaluated 105971 tokens in 202.1s (524.29 tokens/s)
Loss (base 2): 4.6053, Perplexity: 24.34
tensor(4.6053) tensor(24.3408)
```