"""CLIP backbone for visual and textual feature extraction."""

import torch
import torch.nn as nn
import open_clip


class CLIPBackbone(nn.Module):
    """Wraps an OpenCLIP model for dual-encoder feature extraction.

    Extracts visual features from images and textual features from tokenized
    text prompts, with optional projection to a shared embedding space.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        proj_dim: int = 256,
        freeze_visual: bool = True,
        freeze_text: bool = True,
    ):
        super().__init__()
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.clip_dim = self.clip_model.visual.output_dim
        self.proj_dim = proj_dim

        if freeze_visual:
            for p in self.clip_model.visual.parameters():
                p.requires_grad = False

        if freeze_text:
            for p in self.clip_model.transformer.parameters():
                p.requires_grad = False
            for p in self.clip_model.token_embedding.parameters():
                p.requires_grad = False
            if hasattr(self.clip_model, "positional_embedding"):
                self.clip_model.positional_embedding.requires_grad = False
            if hasattr(self.clip_model, "text_projection"):
                self.clip_model.text_projection.requires_grad = False
            if hasattr(self.clip_model, "ln_final"):
                for p in self.clip_model.ln_final.parameters():
                    p.requires_grad = False

        self.visual_proj = nn.Sequential(
            nn.Linear(self.clip_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.clip_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    @torch.no_grad()
    def encode_image_raw(self, images: torch.Tensor) -> torch.Tensor:
        """Extract raw CLIP visual features without projection."""
        return self.clip_model.encode_image(images).float()

    @torch.no_grad()
    def encode_text_raw(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Extract raw CLIP text features without projection."""
        return self.clip_model.encode_text(text_tokens).float()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Extract projected visual features."""
        with torch.no_grad():
            raw = self.clip_model.encode_image(images).float()
        return self.visual_proj(raw)

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Extract projected text features."""
        with torch.no_grad():
            raw = self.clip_model.encode_text(text_tokens).float()
        return self.text_proj(raw)

    def tokenize(self, texts: list) -> torch.Tensor:
        """Tokenize a list of text strings."""
        return self.tokenizer(texts)

    def forward(
        self, images: torch.Tensor, text_tokens: torch.Tensor = None
    ) -> dict:
        """Forward pass returning projected visual (and optional text) features."""
        v_feat = self.encode_image(images)
        out = {"visual": v_feat}
        if text_tokens is not None:
            t_feat = self.encode_text(text_tokens)
            out["textual"] = t_feat
        return out
