# MIMO hetero training sbatches

| Script | Nodes | Layout | GBS | Purpose |
|---|---|---|---|---|
| sbatch_hetero_parity_gbs192.sh | 9 | 1 enc + 8 LLM, TP=2 EP=16 | 192 | 9n Sanjeev parity test (5000 iters, paired with sbatch_sanjeev_parity_gbs192.sh) |
| sbatch_hetero_prod_gbs768_33n_ep8.sh | 33 | 1 enc + 32 LLM, TP=2 EP=8 | 768 | 33n production |
| sbatch_hetero_prod_gbs768_68n_ep8.sh | 68 | 4 enc + 64 LLM, TP=2 EP=8 | 768 | 68n production |
| sbatch_hetero_prod_gbs768_100n.sh    | 100 | 4 enc + 96 LLM, TP=2 EP=8 | 768 | 100n production |

Production sbatches use Sanjeev's WSD schedule (`TRAIN_SAMPLES=122070313`, `LR_WARMUP_SAMPLES=1024000`, `LR_WSD_DECAY_SAMPLES=18310547`) with EP=8 (vs Sanjeev's EP=16), no MTP, no force-LB. Load LLM weights via `--load-nemotron-checkpoint` from sasatheesh's `iter_0001000`.

Launch: `sbatch examples/mimo/scripts/<script>.sh`
