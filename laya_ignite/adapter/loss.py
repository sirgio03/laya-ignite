"""
Multi-task loss functions for Laya decision heads.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class DecisionHeadLoss(nn.Module):
    """Computes combined multi-task loss for choice, score, and noul answers."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, Any],
        question_types: Dict[str, str]
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device, requires_grad=True)

        for qid, logits in predictions.items():
            if qid not in targets:
                continue
            qtype = question_types.get(qid, "choice")
            target = targets[qid]

            if qtype == "choice":
                # target is class index (long)
                loss = self.ce(logits, target)
                total_loss = total_loss + loss
            elif qtype == "noul":
                # target is float 0.0 or 1.0
                loss = self.bce(logits.squeeze(-1), target.float())
                total_loss = total_loss + loss
            elif qtype == "score":
                # target is continuous or ordinal float
                loss = self.mse(logits.squeeze(-1), target.float())
                total_loss = total_loss + loss

        return total_loss
