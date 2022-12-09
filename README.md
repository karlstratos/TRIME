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

# Train

From `train_scripts/wiki103-150M-trime.sh` except I changed "20000,60000" to "2000,6000" for the adaptive stuff and added `--device-id 7 --distributed-world-size 1 --distributed-no-spawn` to use 1 GPU; also `--ce-warmup-epoch` set to -1 to study trime_loss from the beginning
```
python train.py --task language_modeling data-bin/wikitext-2 --save-dir /data/local/TRIME/output/wiki2-150M-trime --arch transformer_lm_wiki103_150M --criterion trime_loss --optimizer adam --adam-betas "(0.9, 0.98)" --weight-decay 0.01 --clip-norm 0.0 --max-update 200000 --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 8000 --warmup-init-lr 1e-07 --max-tokens 9000 --update-freq 2 --tokens-per-sample 150 --seed 1 --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --adaptive-input --tie-adaptive-weights --adaptive-input-cutoff 2000,6000 --adaptive-softmax-cutoff 2000,6000 --knn-keytype last_ffn_input --fp16 --ce-warmup-epoch 3 --required-batch-size-multiple 1 --adaptive-softmax-factor 1 --adaptive-input-factor 1 --device-id 7 --distributed-world-size 1 --distributed-no-spawn  --reset-optimizer --reset-dataloader --ce-warmup-epoch -1
```
