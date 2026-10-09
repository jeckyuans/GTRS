# GTRS / SparseDrive 闭环性能优化

本轮只改变执行效率。冻结 checkpoint、词表、候选选择、FP32、合法输入、nonlinear MPC 和修复后的原图评分保持同一合同。旧部署快照与原工作区保留。所有测量、日志、缓存和闭环输出在 `/ai/ys/alpasim_ws/runs/closedloop_speed_20261009`。

## 实现与来源

- AlpaSim：`/home/ys/alpasim_ws/worktrees/closedloop-speed-v1`，分支 `codex/closedloop-speed-v1`，来源 `codex/navtest-eval-repair@fcc8f91`。
- SparseDrive：`/home/ys/sparsedrive/SparseDriveV2-wt/closedloop-speed-v1`，同名分支，来源 `alpasim-closedloop@25c12f1`。
- GTRS：复制冻结 `driver_snapshot_prefix_cont_v1` 到新工作区的 `e2e_challenge/sample_submission_simscale_navsim_gtrs_frozen`；模型执行文件只有 `model.py` 的缓存修改和新增 `inference_cache.py`。新增目录用于明确绑定已测模型源，未改原快照或权重。
- SparseDrive 接口：复制已验证的 `sparsedrive-v2-interface`，接口数学与时域转换未修改。

共同部分：MTGS 缓存驱逐发生在新 GPU 场景分配之前；正在加载的场景占用容量；同一场景的并发请求共享一个 Future，不同场景仍可并行加载。渲染方法、标定和资产数据未修改。

GTRS 专属：缓存固定词表的 `pos_embed`、可选 encoder 和原始 contiguous repeat 结果；保留注意力输入布局及计算。缓存按 batch 分开，最多四项，不进入 state_dict。

SparseDrive 专属：固定联合轨迹/掩码采用只读 expand，过滤 gather 自行生成输出；删除筛选前的 clone；按原 batch 形状缓存固定 path/velocity 编码。未改变 topk、metric heads 或官方 selector。

两套缓存均在训练、梯度和 autocast 路径绕过，并通过参数版本、地址、device、dtype 和形状变化失效。没有把较大的 batch 或其他 precision 自动设成默认。

## 已完成验证

- 模型缓存 CPU：6 项通过，覆盖更新/重载、dtype、梯度、训练、autocast、batch 与容量。
- Renderer CPU：4 项通过，覆盖驱逐先后顺序、同场景去重、不同场景并行和无效容量。
- 同 checkpoint、8 个固定真实 NAVSIM RGB/ego 帧，batch=1/2/4 共每模型14组对照；GTRS W0、R1 和 SparseDrive 的全部检查输出张量逐位一致，checkpoint 张量未改变。

| 模型 | batch=1 原 forward p50 | 优化后 | 速度比 |
|---|---:|---:|---:|
| GTRS W0 | 267.687 ms | 109.641 ms | 2.4415× |
| GTRS R1 | 282.691 ms | 125.883 ms | 2.2457× |
| SparseDrive V2 | 22.043 ms | 21.652 ms | 1.0180× |

这些是 model-forward 微测量，未包含 RGB 预处理、RPC、渲染、启动和评分。SparseDrive batch=4 的模型峰值 allocated 由 3772742144 降到 3633428992 bytes；不能把它写成整卡/全闭环峰值显存。

机器可读证据：`benchmarks/{gtrs_w0,r1,sparsedrive}.json`，带 tool/args/ckpt/n/dataset/commit/date 及源码/输入 SHA；汇总 `benchmarks/SUMMARY.json`。

## 闭环验证与启动

本轮固定 32 场景、4×4；基线与优化版使用同一修复后 SDK 场景来源、原图评分和 nonlinear MPC。比较器独立检查每场 score、failure_reason、硬失败和资源，并单列原始几何/进度差异。

正式入口在 SparseDrive 优化工作区 `tools/closedloop/prepare_fast_closedloop.py`：显式 `--model {sparsedrive,gtrs_w0,r1}`，默认优化版本；使用 SDK 全量或显式 probe。准备任务后使用 `gpuq submit`。优化版要求对应真实 GPU 输入配对证据仍与源码和输入 SHA 一致。既有 missing299 修复入口的约束保留。

```bash
/ai/ys/alpasim_ws/.venv_sim/bin/python \
  /home/ys/sparsedrive/SparseDriveV2-wt/closedloop-speed-v1/tools/closedloop/prepare_fast_closedloop.py \
  --model r1 --root /ai/ys/alpasim_ws/runs/my_fast_r1 \
  --scope sdk_full --minutes 60
gpuq submit /ai/ys/alpasim_ws/runs/my_fast_r1/gpuq.json
```

SparseDrive 改为 `--model sparsedrive`，保持各自原 interpreter 和 checkpoint。R1/SparseDrive 默认为4路/卡，W0复用共享布局默认值；本轮所有闭环对照显式4×4。新入口将两种模型各自的缓存优化及共同 renderer 优化实际绑定到运行源码。

输出：`closed_loop/{sparse_base32_v1,r1_base32_v1,sparse_opt32_v2,r1_opt32_v2}`。v1 优化项在启动前已取消，不计为实测结果；最终 renderer 使用并行容量预留。

32 场景独立比较已完成，score、failure/status、碰撞/offroad/corridor 硬失败以及全部原始几何/进度子项均逐场一致。两模型各 GPU 的 8GiB 余量与 CPU 峰值 <32 核通过。

| 模型 | 基线32场景 | 优化32场景 | 总耗时变化 |
|---|---:|---:|---:|
| SparseDrive V2 | 265.966 s | 254.686 s | 减少4.24% |
| GTRS R1 | 251.561 s | 258.804 s | 增加2.88% |

GTRS 的微测量与这次总耗时不能混写：总时长包含启动、场景装载及尾部。随后冻结均匀抽取的64个原1186可评分场景，使用相同4×4配置进行更长配对；没有按模型分数选场景。64基线实测311.537s，优化版283.751s，减少8.919%，速度比1.09793×；独立逐场score/硬失败完全一致，CPU峰值19.69/22.22核，GPU余量均通过。

64场景有1场的3项原始子指标不同，不能写成全字段逐bit一致：`2021.05.25.15.59.03_veh-30_04027_04200-46ce401b30ac56e9` 的 `dist_to_gt_trajectory` 0.405526→0.367717、`lateral_dist_to_gt_trajectory` 0.337288→0.367717、`progress_clipped_rel` 1.056086→1.033501。其官方 `progress_score`、score和所有硬事件均一致。动态batch和原运行时数值路径可能参与差异，原因尚未定位；保留完整原始值及ASL，不据此声明全量成功/进度/安全门通过。

结果在 `CLOSED_LOOP_COMPARISON.json` 和后续 `CLOSED_LOOP_COMPARISON64.json`。小集不能作为全 SDK 性能或更宽布局显存的证明；R1/SparseDrive 的4×5不自动启用。初版精确准备与运行文件另存于 `preparation_sources_v1/`，附SHA256，旧任务与失败/取消状态保留。

正式入口的两份完整SDK任务已实际准备：`prepared_sdk_r1/`、`prepared_sdk_sparse/`，各1485场景；各自 `run_alpasim.py pre` 均通过完整场景/资产/源码/配置检查。这两份仅作为可执行的全量任务合同，未提交GPU；本轮所有已提交验证作业均done/rc0，取消的v1从未启动。
