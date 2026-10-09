# Inference copy of SharedVoVDAProgressHead from GTRS navsim/agents/gtrs_dense/da_progress.py.
# Keep module names identical for strict da_head_state_dict loading.
from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from torch import Tensor, nn

PROGRESS_N_PATH = 24
PROGRESS_N_CLASSES = 65
PATH_SCALE = 20.0
STATUS_SCALE = torch.tensor([1., 1., 1., 1., 10., 10., 5., 5.])


def decode_progress(logits: Tensor) -> Tensor:
    centers = (torch.arange(64, device=logits.device, dtype=logits.dtype) + 0.5) * 0.75
    centers = torch.cat([centers, centers.new_full((1,), 48.0)])
    return centers[logits.argmax(dim=-1)]


def _mlp(d_in: int, d: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d), nn.ReLU(inplace=True), nn.LayerNorm(d), nn.Linear(d, d_out))


def path_frame(path: Tensor, mask: Tensor):
    xy = path * mask[..., None]
    prev = torch.cat([xy.new_zeros(xy.shape[0], 1, 2), xy[:, :-1]], dim=1)
    nxt = torch.cat([xy[:, 1:], xy[:, -1:]], dim=1)
    nxt_valid = torch.cat([mask[:, 1:], mask.new_zeros(mask.shape[0], 1)], dim=1)
    ahead = torch.where(nxt_valid[..., None] > 0.5, nxt, xy)
    tangent = ahead - prev
    norm = torch.linalg.norm(tangent, dim=-1, keepdim=True)
    tangent = torch.where(norm > 1e-3, tangent / norm.clamp_min(1e-3),
                          torch.tensor([1.0, 0.0], device=xy.device, dtype=xy.dtype))
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    return tangent, normal


class SharedVoVDAProgressHead(nn.Module):
    """Path-conditioned sparse sampling of the scorer's *same* VoV stitched feature grid.

    ``grid`` is the pre-keyval 16 x 64 VoV feature, not a second image backbone.  References
    and offsets are learned from each predicted route point; the sampler never sees expert
    route points, future images, or progress labels.  The stitched grid has no per-camera
    calibration at deployment, so these are learned image locations rather than metric
    projections.  RGB alignment is an empirical acceptance gate for this fast M1 variant.
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, n_points: int = 4,
                 n_rounds: int = 2):
        super().__init__()
        self.d_model, self.n_points = d_model, n_points
        self.status_encoder = _mlp(8, d_model, d_model)
        self.point_encoder = _mlp(6, d_model, d_model)
        self.point_embedding = nn.Parameter(torch.zeros(1, PROGRESS_N_PATH, d_model))
        nn.init.normal_(self.point_embedding, std=0.02)
        self.reference = nn.ModuleList([nn.Linear(d_model, 2) for _ in range(n_rounds)])
        self.offset = nn.ModuleList([nn.Linear(d_model, n_points * 2) for _ in range(n_rounds)])
        self.weight = nn.ModuleList([nn.Linear(d_model, n_points) for _ in range(n_rounds)])
        self.sample_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_rounds)])
        self.sample_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_rounds)])
        self.mix = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model, nhead, 4 * d_model, dropout=0.0, batch_first=True) for _ in range(n_rounds)])
        self.cls_branch = _mlp(d_model, d_model, PROGRESS_N_CLASSES)
        for reference, offset, weight in zip(self.reference, self.offset, self.weight):
            nn.init.zeros_(reference.weight)
            nn.init.zeros_(reference.bias)
            nn.init.zeros_(offset.weight)
            nn.init.zeros_(offset.bias)
            nn.init.zeros_(weight.weight)
            nn.init.zeros_(weight.bias)

    def forward(self, grid: Tensor, path: Tensor, path_mask: Tensor,
                status: Tensor) -> Dict[str, Tensor]:
        if grid.ndim != 4 or grid.shape[1] != self.d_model:
            raise ValueError(f"VoV grid must be (B,{self.d_model},H,W), got {tuple(grid.shape)}")
        if path.shape[1:] != (PROGRESS_N_PATH, 2) or path_mask.shape != path.shape[:2]:
            raise ValueError("predicted path and mask must have shapes (B,24,2) and (B,24)")
        batch = path.shape[0]
        mask = path_mask > 0.5
        xy = path * mask[..., None]
        poly = torch.cat([xy.new_zeros(batch, 1, 2), xy], dim=1)
        arc = torch.cumsum(torch.linalg.norm(poly[:, 1:] - poly[:, :-1], dim=-1), 1) * mask
        tangent, _ = path_frame(xy, mask.to(xy.dtype))
        points = torch.cat([xy / PATH_SCALE, (arc / PATH_SCALE)[..., None],
                            mask[..., None].to(xy.dtype), tangent], dim=-1)
        tokens = torch.cat([
            self.status_encoder(status[:, :8] / STATUS_SCALE.to(status.device, status.dtype))[:, None],
            self.point_encoder(points) + self.point_embedding,
        ], 1)
        padding = torch.cat([mask.new_zeros(batch, 1), ~mask], 1)
        for ref_layer, offset_layer, weight_layer, projection, norm, mixer in zip(
                self.reference, self.offset, self.weight, self.sample_proj, self.sample_norm, self.mix):
            query = tokens[:, 1:]
            # Grid coordinates are normalized to [-1,1].  Offsets remain within roughly
            # one quarter image in each axis at initialization and are learned thereafter.
            ref = ref_layer(query).tanh()[..., None, :]
            offset = offset_layer(query).view(batch, PROGRESS_N_PATH, self.n_points, 2).tanh() * 0.25
            sample_grid = (ref + offset).reshape(batch, PROGRESS_N_PATH * self.n_points, 1, 2)
            sampled = F.grid_sample(grid, sample_grid, mode="bilinear", padding_mode="zeros",
                                    align_corners=False)
            sampled = sampled[..., 0].transpose(1, 2).reshape(batch, PROGRESS_N_PATH,
                                                                self.n_points, self.d_model)
            weights = weight_layer(query).softmax(-1)
            aggregate = (sampled * weights[..., None]).sum(2)
            query = norm(query + projection(aggregate))
            tokens = mixer(torch.cat([tokens[:, :1], query], 1), src_key_padding_mask=padding)
        logits = self.cls_branch(tokens[:, 0]).float()
        return {"progress_logits": logits, "progress_s4": decode_progress(logits.detach())}
