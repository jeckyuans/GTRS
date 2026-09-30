# Training

Before training GTRS variants, make sure that the pretrained vision backbones and simulated ground-truths for trajectory scorers are downloaded according to [install.md](install.md).

## Diffusion Policy (DP)
We provide the training script utilizing three machine nodes (each with 8 NVIDIA A00 GPUS). 
If you plan to train on a single-node machine, set the variable **NUM_NODES** to 1. 
To ensure a comparable number of gradient updates, set **max_epochs** to 17.

Here is the training script with 3 nodes:
```bash
NUM_NODES=3
MASTER_ADDR=MASTER_NODE_IP # your master node ip
NODE_RANK=0 # 0 for the master node, 1 and 2 for other sub-nodes
config="competition_training" # this config uses the entire navtrain dataset for training
experiment_name=train_dp
agent=gtrs_diffusion_policy

# training hyper-parameters
lr=0.0002
bs=22
max_epochs=50

MASTER_PORT=29500 MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${NUM_NODES} NODE_RANK=${NODE_RANK} \
        python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_dense.py \
            --config-name ${config} \
            trainer.params.num_nodes=${NUM_NODES} \
            agent=${agent} \
            experiment_name=${experiment_name} \
            train_test_split=navtrain \
            dataloader.params.batch_size=${bs} \
            ~trainer.params.strategy \
            trainer.params.max_epochs=${max_epochs} \
            trainer.params.precision=32 \
            agent.config.ckpt_path=${experiment_name} \
            agent.lr=${lr} \
            cache_path=null
```
## GTRS-Dense and Hydra-MDP

Here is the training script with 3 nodes:
```bash
NUM_NODES=3
MASTER_ADDR=MASTER_NODE_IP # your master node ip, which can be set to 127.0.0.1 for single-node training 
NODE_RANK=0 # 0 for the master node, 1 and 2 for other sub-nodes
config="competition_training" # this config uses the entire navtrain dataset for training
experiment_name=train_gtrs_dense # this could also be train_hydra_mdp
agent=gtrs_dense_vov # the agent could also be hydra_mdp_vov

# training hyper-parameters
lr=0.0002
bs=22
max_epochs=20

MASTER_PORT=29500 MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${NUM_NODES} NODE_RANK=${NODE_RANK} \
        python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_dense.py \
            --config-name ${config} \
            trainer.params.num_nodes=${NUM_NODES} \
            agent=${agent} \
            experiment_name=${experiment_name} \
            train_test_split=navtrain \
            dataloader.params.batch_size=${bs} \
            ~trainer.params.strategy \
            trainer.params.max_epochs=${max_epochs} \
            trainer.params.precision=32 \
            agent.config.ckpt_path=${experiment_name} \
            agent.lr=${lr} \
            cache_path=null
```


### Bounded batch prefetch queue (optional)

If decoding the gzip cache is slower than a training step, `run_training_dense.py` can serve training batches from a bounded
queue of already decoded batches (`navsim/planning/training/prefetch.py`). It is off by default. Enable it with
`NAVSIM_PREFETCH_QUEUE=1` or `+dataloader.prefetch_queue.enabled=true`:

| Env var / `+dataloader.prefetch_queue.*` | Default | Meaning |
|---|---|---|
| `NAVSIM_PREFETCH_DEPTH` / `depth` | 8 | decoded batches queued per rank; the first step of each epoch waits until the queue is full |
| `NAVSIM_PREFETCH_RAM_GB` / `ram_budget_gb` | 20 | host-RAM budget for the queue **per rank**; the depth shrinks if `depth x largest batch` exceeds it (`<= 0` disables the cap) |
| `NAVSIM_PREFETCH_DECODE_THREADS` / `decode_threads` | 1 | threads per worker process decoding the samples of one batch concurrently |

With the queue on, `prefetch_factor` is forced to 1, so host memory per rank is about
`(depth + num_workers + 1) x batch size in bytes`. Decoders keep up when `GPU_NUM x num_workers x decode_threads / sample decode time`
exceeds `GPU_NUM x batch_size / step time`. Keep `GPU_NUM x num_workers` within the CPU cores and raise `decode_threads` rather than
the number of worker processes. The validation DataLoader is built through a LightningDataModule with `persistent_workers=False`,
so its workers and batches exist only while validation runs.

## GTRS-Aug

The training hyper-parameters of GTRS-Aug is slightly different from the previous models, but consistent with [DriveSuprim](https://www.arxiv.org/abs/2506.06659):
```bash
NUM_NODES=1
MASTER_ADDR=MASTER_NODE_IP # your master node ip, which can be set to 127.0.0.1 for single-node training 
NODE_RANK=0 # 0 for the master node, 1 and 2 for other sub-nodes
config="competition_training" # this config uses the entire navtrain dataset for training
experiment_name=train_gtrs_aug
agent=gtrs_aug_vov

# training hyper-parameters
lr=0.000075
bs=8
max_epochs=6

MASTER_PORT=29500 MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${NUM_NODES} NODE_RANK=${NODE_RANK} \
        python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_aug.py \
            --config-name ${config} \
            trainer.params.num_nodes=${NUM_NODES} \
            agent=${agent} \
            experiment_name=${experiment_name} \
            train_test_split=navtrain \
            dataloader.params.batch_size=${bs} \
            ~trainer.params.strategy \
            trainer.params.max_epochs=${max_epochs} \
            trainer.params.precision=32 \
            agent.config.ckpt_path=${experiment_name} \
            agent.lr=${lr} \
            cache_path=null
```

