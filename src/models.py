import os
import sys

from rtdetr.zoo.rtdetr.rtdetr import RTDETR
from rtdetr.nn.backbone.presnet import PResNet
from rtdetr.zoo.rtdetr.hybrid_encoder import HybridEncoder
from rtdetr.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import configs.train_cfg as cfg


def build_rtdetr_model_small(pretrained_backbone: bool = False, num_classes: int = None) -> RTDETR:
    """
    Build the RT-DETR small model architecture (ResNet-18 backbone).
    
    Args:
        pretrained_backbone (bool): Whether to use pretrained weights for the backbone.
        num_classes (int): Number of target classes. Defaults to cfg.num_classes.
        
    Returns:
        RTDETR: The constructed small model.
    """
    if num_classes is None:
        num_classes = cfg.num_classes

    # Get input height and width
    inp_h = cfg.input_height
    inp_w = cfg.input_width

    # Build the backbone
    backbone = PResNet(
        depth=18, variant='d', num_stages=4, return_idx=[1, 2, 3],
        freeze_at=-1, freeze_norm=False, pretrained=pretrained_backbone,
    )

    # Build the encoder
    encoder = HybridEncoder(
        in_channels=[128, 256, 512], feat_strides=[8, 16, 32],
        hidden_dim=256, use_encoder_idx=[2], num_encoder_layers=1,
        nhead=8, dim_feedforward=1024, dropout=0.0, enc_act='gelu',
        expansion=0.5, depth_mult=1, act='silu',
        eval_spatial_size=[inp_h, inp_w],
    )

    # Build the decoder
    decoder = RTDETRTransformer(
        num_classes=num_classes, hidden_dim=256, num_queries=300,
        feat_channels=[256, 256, 256], feat_strides=[8, 16, 32],
        num_levels=3, num_points=[4, 4, 4], nhead=8, num_layers=3,
        dim_feedforward=1024, dropout=0.0, activation='relu',
        num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0,
        eval_spatial_size=[inp_h, inp_w], eval_idx=-1,
        cross_attn_method='default', query_select_method='default',
    )

    # Return the model
    return RTDETR(backbone=backbone, encoder=encoder, decoder=decoder)


def build_rtdetr_model_large(pretrained_backbone: bool = False, num_classes: int = None) -> RTDETR:
    """
    Build the RT-DETR large model architecture (ResNet-50 backbone).
    
    Args:
        pretrained_backbone (bool): Whether to use pretrained weights for the backbone.
        num_classes (int): Number of target classes. Defaults to cfg.num_classes.
        
    Returns:
        RTDETR: The constructed large model.
    """
    if num_classes is None:
        num_classes = cfg.num_classes

    # Get input height and width
    inp_h = cfg.input_height
    inp_w = cfg.input_width

    # Build the backbone
    backbone = PResNet(
        depth=50, variant='d', num_stages=4, return_idx=[1, 2, 3],
        freeze_at=0, freeze_norm=True, pretrained=pretrained_backbone,
    )

    # Build the encoder
    encoder = HybridEncoder(
        in_channels=[512, 1024, 2048], feat_strides=[8, 16, 32],
        hidden_dim=256, use_encoder_idx=[2], num_encoder_layers=1,
        nhead=8, dim_feedforward=1024, dropout=0.0, enc_act='gelu',
        expansion=1.0, depth_mult=1, act='silu',
        eval_spatial_size=[inp_h, inp_w],
    )

    # Build the decoder
    decoder = RTDETRTransformer(
        num_classes=num_classes, hidden_dim=256, num_queries=300,
        feat_channels=[256, 256, 256], feat_strides=[8, 16, 32],
        num_levels=3, num_points=[4, 4, 4], nhead=8, num_layers=6,
        dim_feedforward=1024, dropout=0.0, activation='relu',
        num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0,
        eval_spatial_size=[inp_h, inp_w], eval_idx=-1,
        cross_attn_method='default', query_select_method='default',
    )

    # Return the model
    return RTDETR(backbone=backbone, encoder=encoder, decoder=decoder)


def build_rtdetr_model_xlarge(pretrained_backbone: bool = False, num_classes: int = None) -> RTDETR:
    """
    Build the RT-DETR XLarge model architecture (ResNet-101 backbone).
    
    Args:
        pretrained_backbone (bool): Whether to use pretrained weights for the backbone.
        num_classes (int): Number of target classes. Defaults to cfg.num_classes.
        
    Returns:
        RTDETR: The constructed XLarge model.
    """
    if num_classes is None:
        num_classes = cfg.num_classes

    # Get input height and width
    inp_h = cfg.input_height
    inp_w = cfg.input_width

    # Build the backbone
    backbone = PResNet(
        depth=101, variant='d', num_stages=4, return_idx=[1, 2, 3],
        freeze_at=0, freeze_norm=True, pretrained=pretrained_backbone,
    )

    # Build the encoder
    encoder = HybridEncoder(
        in_channels=[512, 1024, 2048], feat_strides=[8, 16, 32],
        hidden_dim=384, use_encoder_idx=[2], num_encoder_layers=1,
        nhead=8, dim_feedforward=2048, dropout=0.0, enc_act='gelu',
        expansion=1.0, depth_mult=1, act='silu',
        eval_spatial_size=[inp_h, inp_w],
    )

    # Build the decoder
    decoder = RTDETRTransformer(
        num_classes=num_classes, hidden_dim=256, num_queries=300,
        feat_channels=[384, 384, 384], feat_strides=[8, 16, 32],
        num_levels=3, num_points=[4, 4, 4], nhead=8, num_layers=6,
        dim_feedforward=1024, dropout=0.0, activation='relu',
        num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0,
        eval_spatial_size=[inp_h, inp_w], eval_idx=-1,
        cross_attn_method='default', query_select_method='default',
    )

    # Return the model
    return RTDETR(backbone=backbone, encoder=encoder, decoder=decoder)


def build_rtdetr_model(variant: str = "small", pretrained_backbone: bool = False, num_classes: int = None) -> RTDETR:
    """
    Build the RT-DETR model architecture based on configuration.
    
    Args:
        variant (str): The RT-DETR variant to build ('small', 'large', 'xlarge').
        pretrained_backbone (bool): Whether to use pretrained weights for the backbone.
        num_classes (int): Number of target classes. Defaults to cfg.num_classes.
        
    Returns:
        RTDETR: The constructed model (small, large, or xlarge based on config).
    """
    variant = variant.lower()
    if variant == "small":
        return build_rtdetr_model_small(pretrained_backbone=pretrained_backbone, num_classes=num_classes)
    elif variant == "large":
        return build_rtdetr_model_large(pretrained_backbone=pretrained_backbone, num_classes=num_classes)
    elif variant == "xlarge":
        return build_rtdetr_model_xlarge(pretrained_backbone=pretrained_backbone, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported RT-DETR variant: {variant}. Choose 'small', 'large', or 'xlarge'.")
