"""
Knowledge Distillation loss functions.

This module implements various KD losses:
- KD1: Logit-based distillation (soft labels with temperature)
- KD2: Sequence-level distillation (teacher-generated targets)
- KD3: Feature-based distillation (hidden state matching)

Corresponds to thesis Section 3.7 - Knowledge Distillation Methods.
"""

from typing import Optional, List, Dict, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetLoss(nn.Module):
    """
    Soft-target distillation loss (KD1).
    
    Loss = alpha * CE(y, p_s) + (1 - alpha) * T^2 * KL(softmax(z_t/T), softmax(z_s/T))
    
    where:
    - y: hard labels
    - p_s: student predictions
    - z_t: teacher logits
    - z_s: student logits
    - T: temperature
    - alpha: weight for hard label loss
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        reduction: str = "mean"
    ):
        """
        Initialize soft target loss.
        
        Args:
            temperature: Softmax temperature for soft targets
            alpha: Weight for hard label CE loss (1-alpha for KL loss)
            reduction: Loss reduction method
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction, ignore_index=-100)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute soft target distillation loss.
        
        Args:
            student_logits: Student model logits [batch, seq, vocab] or [batch, num_classes]
            teacher_logits: Teacher model logits (same shape)
            labels: Optional hard labels for CE loss
            
        Returns:
            Combined loss
        """
        # Soft target loss (KL divergence)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Scale by T^2 as per Hinton et al.
        kl_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Hard label loss
        if labels is not None and self.alpha > 0:
            # Reshape for CE loss if needed
            if len(student_logits.shape) == 3:
                # [batch, seq, vocab] -> [batch*seq, vocab]
                batch_size, seq_len, vocab_size = student_logits.shape
                student_flat = student_logits.view(-1, vocab_size)
                labels_flat = labels.view(-1)
                ce_loss = self.ce_loss(student_flat, labels_flat)
            else:
                ce_loss = self.ce_loss(student_logits, labels)
            
            total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        else:
            total_loss = kl_loss
        
        return total_loss


class SequenceKDLoss(nn.Module):
    """
    Sequence-level distillation loss (KD2).
    
    The student is trained to match teacher-generated sequences
    using standard language modeling loss.
    """
    
    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean"
    ):
        """
        Initialize sequence KD loss.
        
        Args:
            ignore_index: Index to ignore in loss computation
            reduction: Loss reduction method
        """
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction
        )
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sequence-level distillation loss.
        
        Args:
            student_logits: Student logits [batch, seq, vocab]
            teacher_token_ids: Teacher-generated token IDs [batch, seq]
            
        Returns:
            Cross-entropy loss
        """
        # Shift for next-token prediction
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = teacher_token_ids[..., 1:].contiguous()
        
        # Flatten
        vocab_size = shift_logits.size(-1)
        loss = self.ce_loss(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        )
        
        return loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature-based distillation loss (KD3).
    
    Matches intermediate hidden states between teacher and student
    using MSE or cosine similarity loss.
    """
    
    def __init__(
        self,
        layer_mapping: Dict[int, int],
        lambda_feature: float = 0.5,
        loss_type: str = "mse",
        normalize: bool = True,
        projection_dim: Optional[int] = None
    ):
        """
        Initialize feature matching loss.
        
        Args:
            layer_mapping: Mapping from teacher layers to student layers
            lambda_feature: Weight for feature loss
            loss_type: Type of loss (mse, cosine)
            normalize: Whether to normalize hidden states
            projection_dim: If set, project to this dimension before matching
        """
        super().__init__()
        self.layer_mapping = layer_mapping
        self.lambda_feature = lambda_feature
        self.loss_type = loss_type
        self.normalize = normalize
        self.projection_dim = projection_dim
        
        # Projection layers (will be initialized when dimensions are known)
        self.teacher_projections = nn.ModuleDict()
        self.student_projections = nn.ModuleDict()
        
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "cosine":
            self.loss_fn = nn.CosineEmbeddingLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _maybe_create_projection(
        self,
        key: str,
        in_dim: int,
        out_dim: int,
        is_teacher: bool = True
    ):
        """Create projection layer if needed."""
        proj_dict = self.teacher_projections if is_teacher else self.student_projections
        
        if key not in proj_dict:
            proj_dict[key] = nn.Linear(in_dim, out_dim, bias=False)
            # Initialize
            nn.init.xavier_uniform_(proj_dict[key].weight)
    
    def forward(
        self,
        teacher_hiddens: Dict[int, torch.Tensor],
        student_hiddens: Dict[int, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute feature matching loss.
        
        Args:
            teacher_hiddens: Dict mapping layer idx to hidden states [batch, seq, hidden]
            student_hiddens: Dict mapping layer idx to hidden states
            attention_mask: Optional mask [batch, seq]
            
        Returns:
            Feature matching loss
        """
        total_loss = 0.0
        num_layers = 0
        
        for teacher_layer, student_layer in self.layer_mapping.items():
            if teacher_layer not in teacher_hiddens:
                continue
            if student_layer not in student_hiddens:
                continue
            
            t_hidden = teacher_hiddens[teacher_layer]
            s_hidden = student_hiddens[student_layer]
            
            # Project if dimensions differ
            if t_hidden.size(-1) != s_hidden.size(-1):
                target_dim = self.projection_dim or min(t_hidden.size(-1), s_hidden.size(-1))
                
                t_key = f"t_{teacher_layer}"
                s_key = f"s_{student_layer}"
                
                self._maybe_create_projection(t_key, t_hidden.size(-1), target_dim, True)
                self._maybe_create_projection(s_key, s_hidden.size(-1), target_dim, False)
                
                t_hidden = self.teacher_projections[t_key](t_hidden)
                s_hidden = self.student_projections[s_key](s_hidden)
            
            # Normalize if requested
            if self.normalize:
                t_hidden = F.normalize(t_hidden, p=2, dim=-1)
                s_hidden = F.normalize(s_hidden, p=2, dim=-1)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                t_hidden = t_hidden * mask
                s_hidden = s_hidden * mask
            
            # Compute loss
            if self.loss_type == "mse":
                layer_loss = self.loss_fn(s_hidden, t_hidden)
            elif self.loss_type == "cosine":
                # Flatten for cosine loss
                t_flat = t_hidden.view(-1, t_hidden.size(-1))
                s_flat = s_hidden.view(-1, s_hidden.size(-1))
                target = torch.ones(t_flat.size(0), device=t_flat.device)
                layer_loss = self.loss_fn(s_flat, t_flat, target)
            
            total_loss += layer_loss
            num_layers += 1
        
        if num_layers > 0:
            total_loss = total_loss / num_layers
        
        return self.lambda_feature * total_loss


class CombinedKDLoss(nn.Module):
    """
    Combined knowledge distillation loss.
    
    Supports combining multiple KD losses (KD1 + KD3 for example).
    """
    
    def __init__(
        self,
        soft_target_loss: Optional[SoftTargetLoss] = None,
        feature_loss: Optional[FeatureMatchingLoss] = None,
        sequence_loss: Optional[SequenceKDLoss] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize combined loss.
        
        Args:
            soft_target_loss: KD1 loss component
            feature_loss: KD3 loss component
            sequence_loss: KD2 loss component
            weights: Optional weights for each component
        """
        super().__init__()
        self.soft_target_loss = soft_target_loss
        self.feature_loss = feature_loss
        self.sequence_loss = sequence_loss
        self.weights = weights or {"soft_target": 1.0, "feature": 1.0, "sequence": 1.0}
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        teacher_hiddens: Optional[Dict[int, torch.Tensor]] = None,
        student_hiddens: Optional[Dict[int, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        teacher_token_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined KD loss.
        
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        total_loss = 0.0
        components = {}
        
        # KD1: Soft target loss
        if self.soft_target_loss is not None and teacher_logits is not None:
            st_loss = self.soft_target_loss(student_logits, teacher_logits, labels)
            total_loss += self.weights["soft_target"] * st_loss
            components["soft_target"] = st_loss.item()
        
        # KD3: Feature matching loss
        if (self.feature_loss is not None and 
            teacher_hiddens is not None and 
            student_hiddens is not None):
            feat_loss = self.feature_loss(teacher_hiddens, student_hiddens, attention_mask)
            total_loss += self.weights["feature"] * feat_loss
            components["feature"] = feat_loss.item()
        
        # KD2: Sequence loss
        if self.sequence_loss is not None and teacher_token_ids is not None:
            seq_loss = self.sequence_loss(student_logits, teacher_token_ids)
            total_loss += self.weights["sequence"] * seq_loss
            components["sequence"] = seq_loss.item()
        
        return total_loss, components


def create_kd_loss(
    method: str,
    temperature: float = 4.0,
    alpha: float = 0.5,
    lambda_feature: float = 0.5,
    layer_mapping: Optional[Dict[int, int]] = None
) -> nn.Module:
    """
    Factory function to create KD loss based on method name.
    
    Args:
        method: KD method (kd1_logit, kd2_sequence, kd3_feature)
        temperature: Temperature for KD1
        alpha: Alpha for KD1
        lambda_feature: Lambda for KD3
        layer_mapping: Layer mapping for KD3
        
    Returns:
        Appropriate loss module
    """
    if method == "kd1_logit" or method == "KD1_logit":
        return SoftTargetLoss(temperature=temperature, alpha=alpha)
    
    elif method == "kd2_sequence" or method == "KD2_sequence":
        return SequenceKDLoss()
    
    elif method == "kd3_feature" or method == "KD3_feature":
        if layer_mapping is None:
            # Default mapping
            layer_mapping = {4: 2, 8: 4, 12: 6}
        return FeatureMatchingLoss(
            layer_mapping=layer_mapping,
            lambda_feature=lambda_feature
        )
    
    else:
        raise ValueError(f"Unknown KD method: {method}")


class ClassificationKDLoss(nn.Module):
    """
    Specialized KD loss for classification tasks.
    
    Handles the case where we only care about class logits,
    not the full vocabulary.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        temperature: float = 4.0,
        alpha: float = 0.5,
        label_token_ids: Optional[Dict[int, int]] = None
    ):
        """
        Initialize classification KD loss.
        
        Args:
            num_classes: Number of classes
            temperature: Temperature scaling
            alpha: Weight for hard label loss
            label_token_ids: Mapping from class index to token ID
        """
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.label_token_ids = label_token_ids
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    def extract_class_logits(
        self,
        logits: torch.Tensor,
        position: int = -1
    ) -> torch.Tensor:
        """
        Extract class logits from full vocabulary logits.
        
        Args:
            logits: Full logits [batch, seq, vocab]
            position: Position to extract from (default: last token)
            
        Returns:
            Class logits [batch, num_classes]
        """
        if self.label_token_ids is None:
            # Just take first num_classes
            return logits[:, position, :self.num_classes]
        
        # Extract specific token logits
        class_logits = []
        for class_idx in range(self.num_classes):
            token_id = self.label_token_ids[class_idx]
            class_logits.append(logits[:, position, token_id])
        
        return torch.stack(class_logits, dim=-1)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute classification KD loss.
        
        Args:
            student_logits: Student logits (can be full vocab or class-only)
            teacher_logits: Teacher logits
            labels: Hard labels
            
        Returns:
            Combined loss
        """
        # Extract class logits if needed
        if student_logits.size(-1) > self.num_classes:
            student_class_logits = self.extract_class_logits(student_logits)
            teacher_class_logits = self.extract_class_logits(teacher_logits)
        else:
            student_class_logits = student_logits
            teacher_class_logits = teacher_logits
        
        # Soft target loss
        soft_teacher = F.softmax(teacher_class_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_class_logits / self.temperature, dim=-1)
        kl_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Hard label loss
        if labels is not None and self.alpha > 0:
            ce_loss = self.ce_loss(student_class_logits, labels)
            total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        else:
            total_loss = kl_loss
        
        return total_loss
