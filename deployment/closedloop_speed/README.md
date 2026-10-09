# 已验证的 GTRS 闭环推理优化

此目录是实际测试的 `driver_snapshot_prefix_cont_v1` 部署包的优化版本，运行源码逐字节复制，`SOURCE_SHA256.json` 给出完整来源及 SHA256。权重与词表不随代码发布，继续引用 `/ai/ys` 下已冻结资产。

- 缓存固定词表的编码和原始连续 batch 布局；不缓存图像、状态或模型选择结果。
- 梯度、训练和 autocast 路径绕过缓存；权重更新、重载、device/dtype/shape 变化使缓存失效。
- 共享 AlpaSim renderer 改动在 `scripts/closedloop/patches/alpasim-mtgs-renderer-cache.patch`。其原始 servicer blob 为 `60e64eb1c04449d49a8344def59c9cb1d5829569`，与 AlpaSim `cd713e0` 和实测修复源 `fcc8f91` 的该文件相同。先 `git apply --check`，匹配后再应用；不自动覆盖其他模拟器修改。

闭环启动时将 `PYTHONPATH` / driver sample root 指向本目录，其余已冻结 checkpoint、选择参数、控制器、数据与评分合同保持绑定。完整任务的本机准备入口为 SparseDrive 配套 PR 的 `tools/closedloop/prepare_fast_closedloop.py`；迁移绝对路径后必须重新生成源码 hash 和输入配对凭据，不能沿用旧路径的 ready 文件。

验证与限制见 `docs/CLOSEDLOOP_SPEED_20261009.md` 和 `docs/closedloop_speed_20261009/`。64场景 R1 用时311.537→283.751秒，逐场score和硬失败一致；1场3项原始几何/进度值不同，完整原值保留。未重新测完整SDK运行时间，未宣称模型质量或安全晋升。
