# 假设节点1保存在: /path/A/dist_ckpt/ (包含 __00_16.distcp ~ __07_16.distcp)
# 假设节点2保存在: /path/B/dist_ckpt/ (包含 __08_16.distcp ~ __15_16.distcp)

# 选择一个目标目录（比如 A）
TARGET_DIR="/apdcephfs_tj5/share_303570626/xianyihe/verl/checkpoints/verl_grpo_miou2k_stage2_rollout16_qae_nocot/qwen3_vl_8b_grpo_highclip0_28_nocot_qae07/global_step_30/actor/dist_ckpt"
SOURCE_DIR="/jizhi/jizhi2/worker/trainer/checkpoints/verl_grpo_miou2k_stage2_rollout16_qae_nocot/qwen3_vl_8b_grpo_highclip0_28_nocot_qae07/global_step_30/actor/dist_ckpt"

# 复制缺失的 distcp 文件
cp ${SOURCE_DIR}/__*.distcp ${TARGET_DIR}/

# 注意：common.pt 和 metadata.json 只需要保留一份（两个节点的应该是相同的）
