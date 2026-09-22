# -*- coding: utf-8 -*-
"""
Top-3 collision candidate colorizer (TORCH ONLY)
SMPL 24-part skinning -> 19-part merged visualization

- 입력: top3 human part indices (19-part 기준)
- 출력: top3 overlay용 vertex color / alpha
- top1 / top2 / top3 서로 다른 색상
- normal-shading mesh 위에 overlay 하기 위한 용도
"""

import torch
import numpy as np


class CollisionTop3Colorizer19:
    """
    19개 파트 기준 top3 collision candidate를 받아
    vertex overlay color / alpha를 GPU(torch)에서 생성

    - SMPL 24-part skinning -> 19-part 병합
    - top1 / top2 / top3 rank별 색상 구분
    - normal-shading 위에 overlay 하기 위한 color+alpha 분리 방식
    """

    PART24_TO_PART19 = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 8,
        9: 8,
        7: 6,
        10: 6,
        8: 7,
        11: 7,
        12: 9,
        13: 10,
        14: 11,
        15: 12,
        16: 13,
        17: 14,
        18: 15,
        19: 16,
        20: 17,
        22: 17,
        21: 18,
        23: 18,
    }

    def __init__(
        self,
        vertex_weights_24: np.ndarray,
        device: torch.device,
        eps: float = 1e-6,
        min_weight: float = 0.08,
        mask_gamma: float = 2.5,
        top_colors=None,
    ):
        self.device = device
        self.eps = eps
        self.min_weight = float(min_weight)
        self.mask_gamma = float(mask_gamma)

        self.vertex_weights_19 = self._build_vertex_weights_19(
            vertex_weights_24
        ).to(device)

        if top_colors is None:
            top_colors = [
                (1.00, 0.15, 0.15),  # top1 red
                (1.00, 0.78, 0.12),  # top2 yellow
                (0.15, 0.35, 1.00),  # top3 blue
            ]

        self.top_colors = torch.tensor(
            top_colors, dtype=torch.float32, device=device
        )  # (3,3)

    def _build_vertex_weights_19(self, vertex_weights_24: np.ndarray) -> torch.Tensor:
        V = vertex_weights_24.shape[0]
        w24 = torch.from_numpy(vertex_weights_24).float()

        w19 = torch.zeros((V, 19), dtype=torch.float32)
        for j24, j19 in self.PART24_TO_PART19.items():
            w19[:, j19] += w24[:, j24]

        w19 = w19 / (w19.sum(dim=1, keepdim=True) + self.eps)
        return w19

    def _part_mask(self, part_idx: int) -> torch.Tensor:
        if part_idx < 0 or part_idx >= 19:
            raise ValueError(f"part_idx must be in [0,18], got {part_idx}")

        w = self.vertex_weights_19[:, part_idx].clone()  # (6890,)

        # 작은 skinning weight 제거
        w = torch.where(w >= self.min_weight, w, torch.zeros_like(w))

        # normalize
        w_max = torch.max(w)
        if w_max <= self.eps:
            return torch.zeros((w.shape[0], 1), dtype=torch.float32, device=self.device)

        w = w / (w_max + self.eps)

        # 경계 sharpen
        w = torch.pow(w, self.mask_gamma)

        return w.unsqueeze(1)  # (6890,1)

    def get_vertex_overlay(self, top3_indices):
        """
        top3 overlay용 color + alpha 생성

        Parameters
        ----------
        top3_indices : list/tuple, 길이 1~3

        Returns
        -------
        color_map : (6890,3) torch
        alpha_map : (6890,1) torch
        """
        if top3_indices is None:
            V = self.vertex_weights_19.shape[0]
            zero_color = torch.zeros((V, 3), dtype=torch.float32, device=self.device)
            zero_alpha = torch.zeros((V, 1), dtype=torch.float32, device=self.device)
            return zero_color, zero_alpha

        top3_indices = list(top3_indices[:3])

        V = self.vertex_weights_19.shape[0]
        color_acc = torch.zeros((V, 3), dtype=torch.float32, device=self.device)
        alpha_acc = torch.zeros((V, 1), dtype=torch.float32, device=self.device)

        for rank, part_idx in enumerate(top3_indices):
            if part_idx is None:
                continue

            part_idx = int(part_idx)
            if part_idx < 0 or part_idx >= 19:
                continue

            mask = self._part_mask(part_idx)          # (V,1)
            color = self.top_colors[rank].view(1, 3) # (1,3)

            color_acc = color_acc + mask * color
            alpha_acc = torch.maximum(alpha_acc, mask)

        # color 정규화: alpha 있는 곳만 나눔
        color_map = torch.where(
            alpha_acc > self.eps,
            color_acc / (alpha_acc + self.eps),
            torch.zeros_like(color_acc)
        )

        return color_map.contiguous(), alpha_acc.contiguous()

    def get_vertex_colors(self, top3_indices, base_vertex_colors=None):
        """
        기존 호환용 함수
        - base가 있으면 overlay 합성 결과 반환
        - base가 없으면 검정 배경 위 top3 색 반환
        """
        color_map, alpha_map = self.get_vertex_overlay(top3_indices)

        if base_vertex_colors is None:
            base = torch.zeros_like(color_map)
        else:
            base = base_vertex_colors.to(self.device).clone()

        out = base * (1.0 - alpha_map) + color_map * alpha_map
        return out.contiguous()

    def get_part_masks(self, top3_indices):
        masks = []
        for part_idx in top3_indices[:3]:
            masks.append(self._part_mask(int(part_idx)))
        return masks