from .base_cfg import *

# ─────────────────────────────────────────────────────────────────────────────
# Export and PTQ Settings
# ─────────────────────────────────────────────────────────────────────────────
export_config = {
    "opset_version": 17,            # RT-DETR needs >=16 for Deformable Attention; torch 2.8 emits <=20
    "dynamic_axes": False,          # Fixed batch size 1 is usually best for edge deployment
    "batch_size": 1,                # Static batch baked into the ONNX graph
    "simplify": True,               # Run onnxslim on YOLO exports
    "workspace_gb": 4,              # TensorRT builder workspace limit

    # Final artifact evaluation. 0 means the full validation set, matching the
    # default scope of evaluate.py. The INT8 accuracy gate remains a fast,
    # evenly sampled subset controlled by accuracy_validation_images.
    "evaluation_images": 0,

    # TensorRT 11 is strongly-typed: it removed the FP16/INT8 builder flags and
    # the IInt8Calibrator interface, so every precision must already be encoded
    # in the ONNX graph. FP16 engines are therefore built from an FP16-converted
    # ONNX. Ops listed here stay in FP32 during that conversion.
    "fp16_op_block_list": [],
}

# PTQ settings shared by both architectures. Each architecture below starts
# from these and overrides what it needs, so the per-architecture dicts are
# complete and call sites never need a fallback.
_ptq_defaults = {
    # Calibration is part of model preparation, so use representative training
    # data and reserve validation data exclusively for the accuracy gate.
    #
    # Deliberately the full manifest rather than the budgeted one. A calibrator
    # runs images through the graph to collect activation ranges and never opens
    # an annotation, so an annotation budget has nothing to say about it -- the
    # same reason the teacher cache and teacher-scored validation use the full
    # pools. Following the budget would have drawn each arm's `num_calibration_
    # images` from a different pool, so two INT8 engines differing only in
    # training budget would also differ in what they were calibrated on, and any
    # quantization comparison between them would carry that confound for nothing.
    "calibration_annotations": full_train_json,
    "num_calibration_images": 300,
    "calibration_batch_size": 1,    # Must match the static ONNX batch
    "calibration_seed": 42,

    # "percentile" clips the long tail of the activation distribution and is
    # worth roughly 2 mAP over "minmax" on YOLO26n, which has few channels per
    # layer and so is very outlier-sensitive. "mse" is the same idea without a
    # fixed clipping fraction: it picks each threshold by minimising the INT8
    # quantization error over the tensor's own histogram, so how much tail it
    # keeps follows the shape of the distribution. Worth trying when a tensor's
    # tail is heavy enough that 99.999% is still too generous, or light enough
    # that it clips real signal. Implemented in export.py -- ONNX Runtime has
    # no MSE calibrator of its own.
    #
    # Memory note: "minmax" is the only ONNX Runtime calibrator that is bounded
    # by construction (it folds ReduceMin/ReduceMax into the graph and returns
    # scalars). "entropy", "percentile" and "mse" retain every intermediate
    # activation for every calibration image before building their histogram,
    # which needs >100 GB on RT-DETR at 736x1280. Those three are therefore
    # driven in chunks of `calibration_chunk_size` images so the collector
    # merges as it goes.
    "calibration_method": "percentile",  # minmax | entropy | percentile | mse
    "calibration_chunk_size": 8,
    "calibration_cache_dir": calibration_cache_dir,

    # An INT8 artifact is published only when it stays within these absolute
    # mAP drops on an evenly sampled validation subset.
    "accuracy_validation_images": 100,
    "accuracy_num_workers": 2,
    "max_map50_drop": 0.03,
    "max_map50_95_drop": 0.02,

    # PyTorch -> FP32 ONNX numerical parity limits on a real validation image.
    "fp32_logits_max_abs_error": 0.05,
    "fp32_boxes_max_abs_error": 0.005,
}

# ── INT8 coverage ────────────────────────────────────────────────────────────
# `quantize_ops` maps an ONNX op type to how much of it to quantize:
#
#   "<scope>"  quantize the op wherever the named region matches
#   True       shorthand for the "all" scope
#   False      leave this op type in floating point
#
# An op type takes a scope string only when it occurs in more than one region
# of that architecture's graph; where it occurs in just one, True/False says
# everything and the scope would be noise. Each entry below documents its
# alternatives with the node counts they select.
#
# Scopes are regions of the graph, defined per architecture in export.py
# (RTDETR_SCOPES / YOLO26_SCOPES). Anything matching `exclude_patterns` is vetoed
# afterwards regardless of scope.
#
# Explicit Q/DQ nodes remain in the ONNX graph because TensorRT uses them to
# encode INT8 tensor scales; TensorRT normally fuses them into its plan.

yolo_ptq_config = {
    **_ptq_defaults,

    # YOLO26 is not a flat CNN (has backbone, neck/encoder, and head/decoder).
    # Quantization can be scoped per op type. By default, keeping Conv everywhere
    # except the Detect head (via exclude_patterns below) is safe.
    #
    # Counts on the yolo26n graph at export resolution (384 nodes): 102 Conv,
    # 24 Concat, 21 Add, 12 Split, 3 MaxPool, 2 Resize.
    #
    # Similar to RT-DETR, scopes include: "all", "backbone", "encoder",
    # "decoder", "backbone_encoder", "backbone_late", "backbone_last_stage"
    "quantize_ops": {
        "Conv": "all",  # (Can also be set to a specific scope like "backbone")
        "Concat": True,
        "MaxPool": True,
        "AveragePool": True,
        "Resize": True,
        "Add": True,

        # Split is the fan-out at the head of every C3k2/C2f block. Leaving it
        # in FP32 does not merely skip one op: it cuts the INT8 region in half
        # at the centre of *every* block, because cv1 feeds Split and Split
        # feeds both the bottleneck branch and the final Concat. Traced on the
        # published int8 graph, that accounted for 35 of 49 precision-crossing
        # dataflow edges -- five per block across model.2/4/6/8/13/16/19 --
        # each one an INT8->FP32->INT8 round trip through an op that computes
        # nothing and only slices a tensor.
        #
        # Unlike Concat and Add above, Split has a purpose-built quantizer in
        # ONNX Runtime 1.27 (QDQSplit), and it propagates the input's scale to
        # every output via `quantize_output_same_as_input`. The outputs are
        # literal slices of the input, so sharing its scale is exact: no extra
        # calibration tensor, and no accuracy cost by construction. That makes
        # it a safer entry than the two generic-QDQ ops already enabled here,
        # and it should be the last one relaxed if a gate ever fails.
        #
        # Measured, A/B against the identical build with Split off (RTX 3090,
        # 5 trials x 300 iterations, engine execution only, no pre/post):
        #
        #   Split off   2.415 ms   414 FPS   520 engine layers, 111 reformat
        #   Split on    2.342 ms   427 FPS   504 engine layers, 102 reformat
        #
        # Accuracy is unchanged to the third decimal (INT8 gate mAP@0.5:0.95
        # 0.2485 both ways), confirming the scale propagation is exact.
        #
        # So it is a real but small win: 3.0% engine time for 9 fewer reformat
        # layers. Note the ONNX-level and engine-level pictures disagree about
        # how big this is -- quantizing Split removes 35 of the 49 precision
        # crossings in the ONNX dataflow graph, but only 9 of 111 reformat
        # layers in the built engine, because TensorRT was already fusing most
        # of those Q/DQ pairs away. Graph-level crossing counts are a guide to
        # where to look, not a prediction of engine time; measure the engine.
        "Split": True,
    },

    # Substrings matched against ONNX node names; every match stays in FP32.
    # "/model.23/" is the YOLO26 Detect head, whose final 1x1 convolutions emit
    # box distances and class logits directly. Those two tensors have a far
    # wider dynamic range than any feature map, so a shared INT8 activation
    # scale costs several mAP points for a negligible latency saving.
    # Layers 10 and 22 contain the C2PSA (Partial Self-Attention) blocks.
    "exclude_patterns": ["/model.23/", "/model.10/", "/model.22/"],

    # "head_fp32" applies exclude_patterns above; "all_conv" ignores them and
    # quantizes the Detect head too (CLI alias: "e2e").
    "int8_profile": "head_fp32",  # head_fp32 | all_conv
}

rtdetr_ptq_config = {
    **_ptq_defaults,

    # Node counts measured on the r50vd graph at 736x1280 (1114 nodes total).
    # Selecting 125 nodes as configured below.
    #
    # Pooling, Resize and Concat only move data: they propagate a scale rather
    # than computing on it, so they cost no accuracy and keep the INT8 region
    # contiguous instead of forcing an INT8->FP->INT8 round trip at every pool,
    # merge and upsample. That is often worth more than extra Conv coverage.
    #
    # Add and Concat have no purpose-built QDQ quantizer in ONNX Runtime 1.27
    # and fall back to a generic one, so turn those off first if a gate fails.
    # The exporter names the next thing to relax on failure.
    "quantize_ops": {
        # 85 Conv, spread over every region, so the scope is what matters most.
        #   "all" 85 (adds the 3 decoder input_proj 1x1) | "backbone_encoder" 82
        #   "backbone" 55 | "encoder" 27 | "backbone_late" 29 (res_layers.2-3)
        #   "backbone_last_stage" 10 (res_layers.3) | "decoder" 3
        "Conv": "backbone_late",

        # MaxPool (1) and AveragePool (3) are backbone pooling. Resize (2) is
        # the encoder's FPN upsample. To avoid isolated INT8 islands, match the
        # scopes of AveragePool with Conv (e.g. "backbone_late").
        # Scopes: "all" (True) | "backbone" | "backbone_late" | "backbone_last_stage"
        "MaxPool": False,
        "AveragePool": "backbone_late",
        "Resize": False,

        # 13 Concat: "encoder" 4 (FPN/PAN merges) | "decoder" 9 (query and
        # denoising assembly, shape plumbing rather than features) | "all" 13.
        "Concat": False,

        # 109 Add: "backbone" 16 (residuals, fusable to Conv+Add+ReLU) |
        #   "encoder" 12 | "backbone_encoder" 28 | "backbone_late" 9 |
        #   "backbone_last_stage" 3 | "decoder" 81 | "all" 109.
        # To avoid INT8 islands, match Add's backbone scope with Conv's.
        # The decoder's are transformer residuals around LayerNorm, which is
        # decomposed in this graph, so they would be isolated INT8 islands.
        "Add": "backbone_late",

        # 102 Gemm (Linear fused by ORT's MatMulAddFusion). 
        # "ffn" 14: linear1/linear2 of the 6 decoder blocks and the AIFI encoder block.
        # "encoder" 6 | "decoder" 96 | "all" 102 -- all three sweep in
        # sampling_offsets and attention_weights (they steer the deformable
        # attention sampling grid) and the bbox/score heads (wide-range
        # logits, and the bbox head compounds over 6 refinement steps).
        "Gemm": "ffn",
    },

    "exclude_patterns": [],
}

# Drop ops explicitly set to False so they do not appear in the final configuration
yolo_ptq_config["quantize_ops"] = {k: v for k, v in yolo_ptq_config["quantize_ops"].items() if v is not False}
rtdetr_ptq_config["quantize_ops"] = {k: v for k, v in rtdetr_ptq_config["quantize_ops"].items() if v is not False}
