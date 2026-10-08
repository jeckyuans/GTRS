# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from navsim.agents.gtrs_dense.candidate_utility import CandidateUtilityHead, candidate_geometry
from navsim.agents.gtrs_dense.hydra_backbone import HydraBackbone
from navsim.agents.gtrs_dense.hydra_config import HydraConfig
from navsim.agents.gtrs_dense.prefix_progress import (
    PrefixProgressHead, prefix_candidate_arcs, prefix_utility_log_score,
)
from navsim.agents.gtrs_dense.spatial_path import SpatialPathHead, front_view_columns
from navsim.agents.transfuser.transfuser_model import AgentHead
from navsim.agents.utils.attn import MemoryEffTransformer
from navsim.agents.utils.nerf import nerf_positional_encoding


class HydraModel(nn.Module):
    def __init__(self, config: HydraConfig):
        super().__init__()

        self._query_splits = [
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = HydraBackbone(config)

        img_num = 2 if config.use_back_view else 1
        self._keyval_embedding = nn.Embedding(
            config.img_vert_anchors * config.img_horz_anchors * img_num, config.tf_d_model
        )  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self.downscale_layer = nn.Conv2d(self._backbone.img_feat_c, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = HydraTrajHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab_path=config.vocab_path,
            config=config
        )
        self._spatial_head = None
        if config.spatial_path:
            if not config.spatial_anchor_path:
                raise ValueError(
                    "spatial_path requires spatial_anchor_path; "
                    "see scripts/inference/frozen_gtrs_paths.sh")
            self._spatial_head = SpatialPathHead(
                d_model=config.tf_d_model,
                d_ffn=config.tf_d_ffn,
                nhead=config.tf_num_head,
                anchor_path=config.spatial_anchor_path,
                in_channels=config.tf_d_model,
            )
        self._candidate_utility_head = None
        if config.candidate_utility:
            if self._spatial_head is None:
                raise ValueError("candidate_utility requires spatial_path=True and its predicted path")
            self._candidate_utility_head = CandidateUtilityHead(
                query_dim=config.tf_d_model,
                hidden_dim=config.candidate_utility_hidden_dim,
            )
        self._prefix_progress_head = None
        if config.prefix_progress:
            if self._candidate_utility_head is None:
                raise ValueError("prefix_progress requires candidate_utility=True")
            self._prefix_progress_head = PrefixProgressHead(
                query_dim=config.tf_d_model, hidden_dim=config.prefix_progress_hidden_dim)


    def img_feat_blc(self, camera_feature):
        img_features = self._backbone(camera_feature)
        img_features = self.downscale_layer(img_features).flatten(-2, -1)
        img_features = img_features.permute(0, 2, 1)
        return img_features


    def evaluate_dp_proposals(self, features, dp_proposals, topk=10, dp_only_inference=False):
        status_feature: torch.Tensor = features["status_feature"][0]
        camera_feature = features["camera_feature"]

        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_encoding = self._status_encoding(status_feature[:, :8])
        else:
            status_encoding = self._status_encoding(status_feature)

        # original
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]
        img_features = self.img_feat_blc(camera_feature)
        if self._config.use_back_view:
            img_features_back = self.img_feat_blc(features["camera_feature_back"])
            img_features = torch.cat([img_features, img_features_back], 1)
        keyval = img_features
        keyval += self._keyval_embedding.weight[None, ...]
        output: Dict[str, torch.Tensor] = {}
        trajectory = self._trajectory_head.eval_dp_proposals(keyval, status_encoding, dp_proposals, topk=topk,
                                                             dp_only_inference=dp_only_inference)
        output.update(trajectory)
        return output

    def forward(self, features: Dict[str, torch.Tensor],
                interpolated_traj=None) -> Dict[str, torch.Tensor]:
        if self._candidate_utility_head is not None:
            return self._forward_frozen_r1(features, interpolated_traj)
        status_feature: torch.Tensor = features["status_feature"][0]
        camera_feature = features["camera_feature"]

        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_encoding = self._status_encoding(status_feature[:, :8])
        else:
            status_encoding = self._status_encoding(status_feature)

        # original
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]
        img_features = self.img_feat_blc(camera_feature)
        if self._config.use_back_view:
            img_features_back = self.img_feat_blc(features["camera_feature_back"])
            img_features = torch.cat([img_features, img_features_back], 1)
        keyval = img_features

        keyval += self._keyval_embedding.weight[None, ...]

        output: Dict[str, torch.Tensor] = {}
        trajectory = self._trajectory_head(keyval, status_encoding, interpolated_traj)
        output.update(trajectory)
        return output

    def _forward_frozen_r1(self, features, interpolated_traj=None):
        """Select the vocabulary row with the frozen R1 score.

        S = logsigmoid(z_route) + logsigmoid(z_NC + d_NC) + logsigmoid(z_DAC + d_DAC)
            + mean(logsigmoid(prefix at 0.5/1/2/4 s)).
        The EP residual is computed and stored, and the original 8-head argmax
        stays in trajectory_scored. trajectory is the S argmax.
        """
        if self.training:
            raise RuntimeError("this submission only runs the frozen R1 selector at inference")
        if self._prefix_progress_head is None:
            raise RuntimeError("frozen R1 selection requires prefix_progress=True")
        with torch.inference_mode():
            output, front = self._trunk_forward(features, interpolated_traj)
            status = features["status_feature"]
            status = status[0] if isinstance(status, (list, tuple)) else status
            self._append_spatial(output, front, status)
            candidates = output["candidate_trajectories"]
            geometry_parts = []
            for start in range(0, candidates.shape[-3], 1024):
                part = (candidates[start:start + 1024] if candidates.ndim == 3
                        else candidates[:, start:start + 1024])
                geometry_parts.append(candidate_geometry(part, output["spatial_path"]))
            geometry = torch.cat(geometry_parts, dim=1)
            base_logits = torch.stack(
                (output["no_at_fault_collisions"],
                 output["drivable_area_compliance"],
                 output["ego_progress"]), dim=-1)
            row_ids = output["candidate_utility_vocab_ids"]
            candidate_features = output["candidate_utility_features"]
            if status.shape[-1] < 6:
                raise ValueError("prefix_progress requires ego status velocity at columns 4:6")
            ego_velocity = status[:, 4:6]
            prefix_candidates = candidates[None].expand(candidate_features.shape[0], -1, -1, -1)
            candidate_arcs = prefix_candidate_arcs(prefix_candidates)
        output["candidate_utility_vocab_ids"] = row_ids
        output["candidate_utility_geometry"] = geometry.clone()
        output["candidate_utility_base_logits"] = base_logits.clone()
        output["candidate_utility_features"] = candidate_features.clone()
        with torch.no_grad():
            output.update(self._candidate_utility_head(
                output["candidate_utility_features"],
                output["candidate_utility_base_logits"],
                output["candidate_utility_geometry"],
            ))
        output["prefix_candidate_arcs"] = candidate_arcs.clone()
        output["prefix_ego_velocity"] = ego_velocity.clone()
        output.update(self._prefix_progress_head(
            output["candidate_utility_features"].detach(),
            output["candidate_utility_geometry"].detach(),
            output["prefix_candidate_arcs"].detach(),
            output["prefix_ego_velocity"].detach(),
        ))
        output["prefix_utility_log_scores"] = prefix_utility_log_score(
            output["candidate_utility_logits"].detach(),
            output["prefix_progress_log_score"],
        )
        index = output["prefix_utility_log_scores"].argmax(dim=1)
        output["trajectory_scored"] = output["trajectory"]
        output["selected_indices_scored"] = output["selected_indices"]
        output["selected_indices"] = output["candidate_utility_vocab_ids"][index]
        output["trajectory"] = self._trajectory_head.vocab[index]
        return output

    def _trunk_forward(self, features, interpolated_traj=None):
        status_feature = features["status_feature"]
        status_feature = status_feature[0] if isinstance(status_feature, (list, tuple)) else status_feature
        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_encoding = self._status_encoding(status_feature[:, :8])
        else:
            status_encoding = self._status_encoding(status_feature)
        camera_feature = features["camera_feature"]
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]
        img_features = self.img_feat_blc(camera_feature)
        if self._config.use_back_view:
            img_features_back = self.img_feat_blc(features["camera_feature_back"])
            img_features = torch.cat([img_features, img_features_back], 1)
        front = self._front_view(img_features)
        keyval = img_features + self._keyval_embedding.weight[None, ...]
        output: Dict[str, torch.Tensor] = {}
        output.update(self._trajectory_head(keyval, status_encoding, interpolated_traj))
        return output, front

    def _front_view(self, img_features):
        height, width = self._config.img_vert_anchors, self._config.img_horz_anchors
        grid = img_features[:, :height * width].reshape(-1, height, width, img_features.shape[-1])
        start, stop = front_view_columns(width)
        return grid[:, :, start:stop].permute(0, 3, 1, 2).contiguous()

    def _append_spatial(self, output, front, status_feature):
        if self._config.spatial_freeze:
            front = front.detach()
        plan_query = output.pop("spatial_plan_query", None)
        if plan_query is not None:
            plan_query = plan_query.detach()
        raw_status = status_feature[:, :8] if status_feature.shape[-1] != 8 else status_feature
        command = raw_status[:, :4].argmax(dim=-1)
        output.update(self._spatial_head(front, plan_query, command))


class HydraTrajHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, vocab_path: str,
                 nhead: int, nlayers: int, config: HydraConfig = None
                 ):
        super().__init__()
        self.config = config
        self._num_poses = num_poses
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), nlayers
        )
        self.vocab = nn.Parameter(
            torch.from_numpy(np.load(vocab_path)),
            requires_grad=False
        )

        self.heads = nn.ModuleDict({
            'no_at_fault_collisions': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'drivable_area_compliance':
                nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
            'time_to_collision_within_bound': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'ego_progress': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'driving_direction_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'lane_keeping': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'traffic_light_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            ),
            'imi': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            )
        })

        self.inference_imi_weight = config.inference_imi_weight
        self.inference_da_weight = config.inference_da_weight
        self.normalize_vocab_pos = config.normalize_vocab_pos
        if self.normalize_vocab_pos:
            self.encoder = MemoryEffTransformer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.0
            )
        self.use_nerf = config.use_nerf

        if self.use_nerf:
            self.pos_embed = nn.Sequential(
                nn.Linear(1040, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_model),
            )
        else:
            self.pos_embed = nn.Sequential(
                nn.Linear(num_poses * 3, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_model),
            )

    def forward(self, bev_feature, status_encoding, interpolated_traj=None) -> Dict[str, torch.Tensor]:
        result = {}
        # vocab: 4096, 40, 3
        # bev_feature: B, 32, C
        # embedded_vocab: B, 4096, C
        vocab = self.vocab.data
        L, HORIZON, _ = vocab.shape
        B = bev_feature.shape[0]
        num_total = vocab.size(0)  # 16384
        if self.training and self.config.vocab_dropout and not self.config.candidate_utility:
            num_select = num_total // 2  # 8192
            indices = torch.randperm(num_total, device=vocab.device)[:num_select]
            vocab = vocab[indices]
            result['dropout_indices'] = indices
            L, HORIZON, _ = vocab.shape
        else:
            result['dropout_indices'] = torch.arange(num_total, device=vocab.device)
        result['trajectory_vocab_dropout'] = vocab
        if self.use_nerf:
            vocab = torch.cat(
                [
                    nerf_positional_encoding(vocab[..., :2]),
                    torch.cos(vocab[..., -1])[..., None],
                    torch.sin(vocab[..., -1])[..., None],
                ], dim=-1
            )

        if self.normalize_vocab_pos:
            embedded_vocab = self.pos_embed(vocab.view(L, -1))[None]
            embedded_vocab = self.encoder(embedded_vocab).repeat(B, 1, 1)
        else:
            embedded_vocab = self.pos_embed(vocab.view(L, -1))[None].repeat(B, 1, 1)
        tr_out = self.transformer(embedded_vocab, bev_feature)
        dist_status = tr_out + status_encoding.unsqueeze(1)
        if self.config.candidate_utility:
            result["candidate_utility_features"] = dist_status
            result["candidate_utility_vocab_ids"] = result["dropout_indices"]

        # selected_indices: B,
        for k, head in self.heads.items():
            result[k] = head(dist_status).squeeze(-1)

        scores = (
                0.03 * result['imi'].softmax(-1).log() +
                0.1 * result['traffic_light_compliance'].sigmoid().log() +
                0.1 * result['no_at_fault_collisions'].sigmoid().log() +
                0.9 * result['drivable_area_compliance'].sigmoid().log() +
                0.2 * result['driving_direction_compliance'].sigmoid().log() +
                6.0 * (7.0 * result['time_to_collision_within_bound'].sigmoid() +
                       7.0 * result['ego_progress'].sigmoid() +
                       3.0 * result['lane_keeping'].sigmoid()
                       ).log()
        )


        selected_indices = scores.argmax(1)
        result["trajectory"] = self.vocab.data[selected_indices]
        result["trajectory_vocab"] = self.vocab.data
        result["selected_indices"] = selected_indices
        if self.config.spatial_path:
            result["candidate_scores"] = scores
            result["candidate_trajectories"] = result["trajectory_vocab_dropout"]
            topk = self.config.spatial_plan_topk
            if topk > 0:
                top = scores.topk(min(topk, scores.shape[1]), dim=1).indices
                result["spatial_plan_query"] = dist_status.gather(
                    1, top[..., None].expand(-1, -1, dist_status.shape[-1]))
        return result

    def eval_dp_proposals(self, bev_feature,
                          status_encoding,
                          dp_proposals,
                          topk=10,
                          dp_only_inference=False) -> Dict[str, torch.Tensor]:
        # vocab: 4096, 40, 3
        # bev_feature: B, 32, C
        # embedded_vocab: B, 4096, C
        vocab = self.vocab.data
        L, HORIZON, TRAJ_DIM = vocab.shape
        B = bev_feature.shape[0]

        NUM_PROPOSALS = dp_proposals.shape[1]
        dp_proposals = dp_proposals.view(B, NUM_PROPOSALS, -1)
        vocab = torch.cat([
            vocab.view(L, -1)[None].repeat(B, 1, 1),
            dp_proposals
        ], 1)

        embedded_vocab = self.pos_embed(vocab)
        embedded_vocab = self.encoder(embedded_vocab)

        tr_out = self.transformer(embedded_vocab, bev_feature)
        dist_status = tr_out + status_encoding.unsqueeze(1)
        result = {}
        # selected_indices: B,
        for k, head in self.heads.items():
            result[k] = head(dist_status).squeeze(-1)
        scene_cnt_tensor = torch.arange(B, device=tr_out.device)

        # only dp: 87 > dp and vocab: 86.6
        scores = (
                         0.01 * result['imi'].softmax(-1).log() +
                         0.1 * result['traffic_light_compliance'].sigmoid().log() +
                         0.5 * result['no_at_fault_collisions'].sigmoid().log() +
                         0.5 * result['drivable_area_compliance'].sigmoid().log() +
                         0.5 * result['driving_direction_compliance'].sigmoid().log() +
                         3.0 * (5.0 * result['time_to_collision_within_bound'].sigmoid() +
                                5.0 * result['ego_progress'].sigmoid() +
                                2.0 * result['lane_keeping'].sigmoid()
                                ).log()
                 )
        if dp_only_inference:
            selected_indices = scores[:, L:].argmax(1)
            result["trajectory"] = dp_proposals[scene_cnt_tensor, selected_indices].view(B, HORIZON, 3)
        else:
            selected_indices = scores.argmax(1)
            result["trajectory"] = vocab[scene_cnt_tensor, selected_indices].view(B, HORIZON, 3)
        result['overall_scores'] = (
                                           1 * result['traffic_light_compliance'].sigmoid() *
                                           1 * result['no_at_fault_collisions'].sigmoid() *
                                           1 * result['drivable_area_compliance'].sigmoid() *
                                           1 * result['driving_direction_compliance'].sigmoid() *
                                           (5.0 * result['time_to_collision_within_bound'].sigmoid() +
                                            5.0 * result['ego_progress'].sigmoid() +
                                            2.0 * result['lane_keeping'].sigmoid()) / 12.0
                                   )[:, L:]
        result['overall_log_scores'] = scores[:, L:]
        _, topk_indices = torch.topk(result['overall_log_scores'], k=topk, dim=1)  # [B, 10]
        result['trajectory_vocab'] = vocab.view(B, NUM_PROPOSALS+L, HORIZON, TRAJ_DIM)
        if not self.training:
            return result

        # rewrite subscore predictions: 16384->16384+top-10
        # this is only for training
        for k in self.heads.keys():
            # for imi we train the model with a vocab 16384+all 100
            if k == 'imi':
                continue
            original_scores = result[k]
            vocab_scores = original_scores[:, :L]
            dp_scores = original_scores[:, L:]

            selected_dp_scores = torch.gather(dp_scores, dim=1, index=topk_indices)  # [B, 10]
            result[k] = torch.cat([vocab_scores, selected_dp_scores], dim=1)  # [B, 16384 + 10]
        return result
