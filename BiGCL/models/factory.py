"""Shared model construction from config dicts."""

import torch

from .bigcl import BiGCL


def build_model_from_config(cfg: dict, device: torch.device) -> BiGCL:
    """Instantiate BiGCL model from a config dict and move to device.

    Args:
        cfg: full config dict with 'model' and 'loss' sections
        device: target device
    Returns:
        BiGCL model on the specified device
    """
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]

    model = BiGCL(
        clip_model_name=model_cfg["clip_model_name"],
        clip_pretrained=model_cfg["clip_pretrained"],
        n_clusters=model_cfg["n_clusters"],
        proj_dim=model_cfg["proj_dim"],
        n_ctx=model_cfg["n_ctx"],
        gnn_layers=model_cfg["gnn_layers"],
        gnn_heads=model_cfg["gnn_heads"],
        gnn_top_k=model_cfg["gnn_top_k"],
        gnn_temperature=model_cfg["gnn_temperature"],
        feat_drop_rate=model_cfg["feat_drop_rate"],
        edge_drop_rate=model_cfg["edge_drop_rate"],
        sinkhorn_iters=model_cfg["sinkhorn_iters"],
        node_temperature=loss_cfg["node_temperature"],
        cross_temperature=loss_cfg["cross_temperature"],
        graph_temperature=loss_cfg["graph_temperature"],
        w_node=loss_cfg["w_node"],
        w_cross=loss_cfg["w_cross"],
        w_graph=loss_cfg["w_graph"],
        w_uniform=loss_cfg["w_uniform"],
        w_entropy=loss_cfg["w_entropy"],
        w_self_train=loss_cfg["w_self_train"],
        use_momentum=model_cfg["use_momentum"],
        base_momentum=model_cfg["base_momentum"],
        dropout=model_cfg["dropout"],
    )
    return model.to(device)
