# -*- coding: utf-8 -*-
"""
Effective mass → vertex color mapper (TORCH ONLY)
24-part SMPL skinning → 19-part merged visualization
"""

import torch
import numpy as np
import matplotlib.cm as cm


class EffectiveMassColorizer19:
    """
    19개 파트 기준 유효질량(Meff)을 받아
    vertex color (6890,3)을 GPU(torch)에서 바로 생성

    - SMPL 24-part skinning → 19-part 병합
    - plasma colormap (torch LUT)
    - 실시간 시각화 최적화
    """

    # --------------------------------------------------
    # 24-part → 19-part mapping
    # --------------------------------------------------
    PART24_TO_PART19 = {
        0: 0,    # pelvis
        1: 1,    # L_hip
        2: 2,    # R_hip
        3: 3,    # spine1
        4: 4,    # L_knee
        5: 5,    # R_knee

        6: 8,    # spine2
        9: 8,    # spine3 → spine2

        7: 6,    # L_ankle
        10: 6,   # L_foot → L_ankle

        8: 7,    # R_ankle
        11: 7,   # R_foot → R_ankle

        12: 9,   # neck

        13: 10,  # L_collar
        14: 11,  # R_collar

        15: 12,  # head

        16: 13,  # L_shoulder
        17: 14,  # R_shoulder

        18: 15,  # L_elbow
        19: 16,  # R_elbow

        20: 17,  # L_wrist
        22: 17,  # L_hand → L_wrist

        21: 18,  # R_wrist
        23: 18,  # R_hand → R_wrist
    }

    # --------------------------------------------------
    # Thresholds (kg)
    # --------------------------------------------------
    FIXED_THRESHOLDS = torch.tensor([
        22.0,   # pelvis
        16.0,   # L_hip
        16.0,   # R_hip
        17.94,  # spine1
        16.0,   # L_knee
        16.0,   # R_knee
        16.0,   # L_ankle (+foot)
        16.0,   # R_ankle (+foot)
        19.29,  # spine2 (+spine3)
        8.0,    # neck
        6.0,    # L_collar
        6.0,    # R_collar
        6.5,    # head
        3.6,    # L_shoulder
        3.6,    # R_shoulder
        1.7,    # L_elbow
        1.7,    # R_elbow
        0.5,    # L_wrist (+hand)
        0.5,    # R_wrist (+hand)
    ], dtype=torch.float32)

    # --------------------------------------------------
    # Init
    # --------------------------------------------------
    def __init__(
        self,
        vertex_weights_24: np.ndarray,  # (6890,24)
        device: torch.device,
        eps: float = 1e-6,
    ):
        """
        Parameters
        ----------
        vertex_weights_24 : (6890,24) numpy
            SMPL skinning weights
        device : torch.device
            cuda or cpu
        eps : float
            numerical stability
        """
        self.device = device
        self.eps = eps

        # ---------------------------
        # Build merged vertex weights
        # ---------------------------
        self.vertex_weights_19 = self._build_vertex_weights_19(
            vertex_weights_24
        ).to(device)  # (6890,19)

        # ---------------------------
        # Threshold
        # ---------------------------
        self.joint_threshold = self.FIXED_THRESHOLDS.to(device)

        # ---------------------------
        # Plasma LUT (256,3)
        # ---------------------------
        self.plasma_lut = self._build_plasma_lut().to(device)

    # --------------------------------------------------
    # 24 → 19 vertex weight merge (torch)
    # --------------------------------------------------
    def _build_vertex_weights_19(self, vertex_weights_24: np.ndarray) -> torch.Tensor:
        V = vertex_weights_24.shape[0]
        w24 = torch.from_numpy(vertex_weights_24).float()

        w19 = torch.zeros((V, 19), dtype=torch.float32)
        for j24, j19 in self.PART24_TO_PART19.items():
            w19[:, j19] += w24[:, j24]

        # normalize
        w19 = w19 / (w19.sum(dim=1, keepdim=True) + self.eps)
        return w19

    # --------------------------------------------------
    # Plasma LUT
    # --------------------------------------------------
    def _build_plasma_lut(self) -> torch.Tensor:
        plasma = cm.get_cmap("plasma")
        lut = plasma(np.linspace(0.0, 1.0, 256))[:, :3]  # RGB
        return torch.from_numpy(lut).float()

    # --------------------------------------------------
    # Meff → ratio
    # --------------------------------------------------
    def compute_excess_ratio(self, meff_19: torch.Tensor) -> torch.Tensor:
        """
        meff_19 : (19,) torch
        """
        ratio = (meff_19 - self.joint_threshold) / (self.joint_threshold + self.eps)
        return torch.clamp(ratio, 0.0, 1.0)

    # --------------------------------------------------
    # ratio → color (torch LUT)
    # --------------------------------------------------
    def ratio_to_part_colors(self, ratio_19: torch.Tensor) -> torch.Tensor:
        """
        (19,) → (19,3)
        """
        idx = (ratio_19 * 255).long().clamp(0, 255)
        return self.plasma_lut[idx]

    # --------------------------------------------------
    # Meff → vertex color (FINAL)
    # --------------------------------------------------
    def get_vertex_colors(self, meff_19: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        meff_19 : (19,) torch, device=self.device

        Returns
        -------
        vertex_colors : (6890,3), float32, contiguous, torch
        """
        ratio_19 = self.compute_excess_ratio(meff_19)
        part_colors = self.ratio_to_part_colors(ratio_19)  # (19,3)

        vertex_colors = self.vertex_weights_19 @ part_colors  # (6890,3)
        return vertex_colors.contiguous()
