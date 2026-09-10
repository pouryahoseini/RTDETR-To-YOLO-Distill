import json
from pathlib import Path
import sys

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import configs.export_cfg as cfg

from evaluate import (  # noqa: E402
    EvalDataset,
    _EvaluationMemoryBenchmark,
    _resolve_evaluation_batch_size,
    decode_yolo_e2e_output,
    eval_collate_fn,
    generate_reports,
    profile_deployment_artifact,
    profile_onnx_model,
    resolve_image_path,
    resolve_profile_onnx,
)
from export import (  # noqa: E402
    RTDETR_SCOPES,
    YOLO26_SCOPES,
    CocoCalibrationDataReader,
    _CALIBRATION_METHODS,
    _ChunkedCalibrationDataReader,
    _MSEHistogramCollector,
    _canonical_scope,
    _clean_scalar_qdq_nodes,
    _relaxation_hint,
    _resolve_quantize_ops,
    _resolve_yolo_int8_profile,
    _selected_nodes,
    _validation_subsample_interval,
    artifact_path,
    ptq_config,
    validate_qdq_for_tensorrt,
)
from trt_export import _check_graph_matches_precision  # noqa: E402


def _save_activation_qdq(path, zero_point=0):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale")
    zp = numpy_helper.from_array(np.array(zero_point, dtype=np.int8), "zero_point")
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "scale", "zero_point"], ["xq"], name="q"),
        helper.make_node("DequantizeLinear", ["xq", "scale", "zero_point"], ["y"], name="dq"),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "qdq", [x], [y], [scale, zp]),
        opset_imports=[helper.make_opsetid("", 16)],
    )
    model.ir_version = 8
    onnx.save(model, path)


def _save_scalar_dq_models(
    source_path,
    quantized_path,
    *,
    quantized_name="offset_quantized",
    include_source=True,
    dynamic_pair=False,
):
    """Write a source graph and its ORT-style scalar-DQ counterpart."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    offset = numpy_helper.from_array(np.array(0.73, dtype=np.float32), "offset")

    if include_source:
        source_nodes = [helper.make_node("Add", ["x", "offset"], ["y"], name="source_add")]
        source_initializers = [offset]
    else:
        source_nodes = [helper.make_node("Identity", ["x"], ["y"], name="source_identity")]
        source_initializers = []
    source = helper.make_model(
        helper.make_graph(source_nodes, "source", [x], [y], source_initializers),
        opset_imports=[helper.make_opsetid("", 16)],
    )
    source.ir_version = 8
    onnx.checker.check_model(source)
    onnx.save(source, source_path)

    scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale")
    zero_point = numpy_helper.from_array(np.array(0, dtype=np.int8), "zero_point")
    initializers = [scale, zero_point]
    nodes = [
        helper.make_node(
            "QuantizeLinear", ["x", "scale", "zero_point"], ["xq"], name="activation_q"
        ),
        helper.make_node(
            "DequantizeLinear", ["xq", "scale", "zero_point"], ["xdq"], name="activation_dq"
        ),
    ]
    if dynamic_pair:
        initializers.append(offset)
        nodes.extend(
            [
                helper.make_node(
                    "QuantizeLinear",
                    ["offset", "scale", "zero_point"],
                    ["offset_q"],
                    name="scalar_q",
                ),
                helper.make_node(
                    "DequantizeLinear",
                    ["offset_q", "scale", "zero_point"],
                    ["offset_dq"],
                    name="scalar_dq",
                ),
            ]
        )
    else:
        initializers.append(
            numpy_helper.from_array(np.array(7, dtype=np.int8), quantized_name)
        )
        # Deliberately unnamed: the cleanup must identify this node by its
        # unique output, not by a potentially missing or duplicate node name.
        nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [quantized_name, "scale", "zero_point"],
                ["offset_dq"],
            )
        )
    nodes.append(helper.make_node("Add", ["xdq", "offset_dq"], ["y"], name="quantized_add"))

    quantized = helper.make_model(
        helper.make_graph(nodes, "quantized", [x], [y], initializers),
        opset_imports=[helper.make_opsetid("", 16)],
    )
    quantized.ir_version = 8
    onnx.checker.check_model(quantized)
    onnx.save(quantized, quantized_path)


def _save_profile_graph(path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info("d", TensorProto.FLOAT, [1, 1, 3, 3])
    weight = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), "w")
    scales = numpy_helper.from_array(np.ones(4, dtype=np.float32), "scales")
    names = [
        "/model/backbone/conv1/Conv",
        "/model/encoder/input_proj/Conv",
        "/model/decoder/input_proj/Conv",
        "/model/backbone/res_layers.3/blocks.2/branch2a/conv/Conv",
    ]
    tensors = ["x", "a", "b", "c", "d"]
    nodes = [
        helper.make_node("Conv", [tensors[i], "w"], [tensors[i + 1]], name=name)
        for i, name in enumerate(names)
    ]
    # One node per non-Conv op type the wider profiles reach for, plus decoder
    # nodes that every profile must leave alone.
    nodes += [
        helper.make_node(
            "MaxPool", ["d"], ["e"], name="/model/backbone/pool/MaxPool", kernel_shape=[1, 1]
        ),
        helper.make_node(
            "AveragePool",
            ["e"],
            ["f"],
            name="/model/backbone/short/AveragePool",
            kernel_shape=[1, 1],
        ),
        helper.make_node(
            "Resize", ["f", "", "scales"], ["g"], name="/model/encoder/upsample/Resize"
        ),
        helper.make_node("Concat", ["g", "g"], ["h"], name="/model/encoder/fpn/Concat", axis=1),
        helper.make_node("Add", ["h", "h"], ["i"], name="/model/backbone/res_layers.1/blocks.0/Add"),
        helper.make_node("Add", ["i", "i"], ["j"], name="/model/decoder/layers.0/Add"),
        helper.make_node(
            "Gemm", ["j", "w"], ["k"], name="/model/decoder/layers.0/linear1/MatMul/MatMulAddFusion"
        ),
        helper.make_node(
            "Gemm", ["k", "w"], ["l"], name="/model/decoder/layers.0/linear2/MatMul/MatMulAddFusion"
        ),
        helper.make_node(
            "Gemm",
            ["l", "w"],
            ["m"],
            name="/model/decoder/layers.0/cross_attn/sampling_offsets/MatMul/MatMulAddFusion",
        ),
        helper.make_node(
            "Gemm", ["m", "w"], ["n"], name="/model/decoder/dec_score_head.0/MatMul/MatMulAddFusion"
        ),
        helper.make_node("MatMul", ["n", "n"], ["o"], name="/model/decoder/layers.0/self_attn/MatMul"),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "profiles", [x], [y], [weight, scales]),
        opset_imports=[helper.make_opsetid("", 16)],
    )
    onnx.save(model, path)


def _scope_nodes(path, quantize_ops, scopes=None):
    spec = _resolve_quantize_ops(quantize_ops, scopes or RTDETR_SCOPES)
    return _selected_nodes(str(path), spec)[0]


# ── Q/DQ validation ──────────────────────────────────────────────────────────
def test_qdq_validator_accepts_symmetric_graph(tmp_path):
    path = tmp_path / "symmetric.onnx"
    _save_activation_qdq(path, zero_point=0)
    assert validate_qdq_for_tensorrt(str(path)) == {
        "quantize_nodes": 1,
        "dequantize_nodes": 1,
    }


def test_qdq_validator_rejects_asymmetric_graph(tmp_path):
    path = tmp_path / "asymmetric.onnx"
    _save_activation_qdq(path, zero_point=-3)
    with pytest.raises(ValueError, match="zero-point 0"):
        validate_qdq_for_tensorrt(str(path))


def test_qdq_validator_rejects_graph_without_qdq_nodes(tmp_path):
    path = tmp_path / "plain.onnx"
    _save_profile_graph(path)
    with pytest.raises(ValueError, match="no Q/DQ nodes"):
        validate_qdq_for_tensorrt(str(path))


# ── Scalar Q/DQ cleanup ───────────────────────────────────────────────────────
def test_scalar_dq_cleanup_restores_source_and_preserves_shared_qdq(tmp_path):
    source_path = tmp_path / "source.onnx"
    quantized_path = tmp_path / "quantized.onnx"
    _save_scalar_dq_models(source_path, quantized_path)

    with pytest.raises(ValueError, match="scalar initializer"):
        validate_qdq_for_tensorrt(str(quantized_path))

    assert _clean_scalar_qdq_nodes(str(quantized_path), str(source_path)) == 1
    cleaned = onnx.load(quantized_path)
    onnx.checker.check_model(cleaned)
    assert validate_qdq_for_tensorrt(str(quantized_path)) == {
        "quantize_nodes": 1,
        "dequantize_nodes": 1,
    }

    initializers = {init.name for init in cleaned.graph.initializer}
    assert {"offset", "scale", "zero_point"} <= initializers
    assert "offset_quantized" not in initializers
    assert next(node for node in cleaned.graph.node if node.name == "quantized_add").input[1] == "offset"

    sample = np.array([[0.2, -0.4]], dtype=np.float32)
    providers = ["CPUExecutionProvider"]
    expected = ort.InferenceSession(str(source_path), providers=providers).run(None, {"x": sample})
    actual = ort.InferenceSession(str(quantized_path), providers=providers).run(None, {"x": sample})
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=1e-7)


def test_scalar_cleanup_does_not_rewrite_dynamic_qdq_pair(tmp_path):
    source_path = tmp_path / "source.onnx"
    quantized_path = tmp_path / "dynamic.onnx"
    _save_scalar_dq_models(source_path, quantized_path, dynamic_pair=True)
    original = quantized_path.read_bytes()

    assert _clean_scalar_qdq_nodes(str(quantized_path), str(source_path)) == 0
    assert quantized_path.read_bytes() == original
    onnx.checker.check_model(onnx.load(quantized_path))
    with pytest.raises(ValueError, match="scalar initializer"):
        validate_qdq_for_tensorrt(str(quantized_path))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"quantized_name": "offset_q"}, "_quantized.*suffix"),
        ({"include_source": False}, "absent"),
    ],
)
def test_scalar_cleanup_fails_closed_without_exact_source(tmp_path, options, message):
    source_path = tmp_path / "source.onnx"
    quantized_path = tmp_path / "quantized.onnx"
    _save_scalar_dq_models(source_path, quantized_path, **options)
    original = quantized_path.read_bytes()

    with pytest.raises(ValueError, match=message):
        _clean_scalar_qdq_nodes(str(quantized_path), str(source_path))

    assert quantized_path.read_bytes() == original
    onnx.checker.check_model(onnx.load(quantized_path))


# ── RT-DETR coverage scopes ──────────────────────────────────────────────────
def test_conv_scopes_select_only_the_named_region(tmp_path):
    path = tmp_path / "profiles.onnx"
    _save_profile_graph(path)
    assert len(_scope_nodes(path, {"Conv": "all"})) == 4
    assert len(_scope_nodes(path, {"Conv": "backbone_encoder"})) == 3
    assert len(_scope_nodes(path, {"Conv": "backbone"})) == 2
    assert len(_scope_nodes(path, {"Conv": "backbone_late"})) == 1
    assert len(_scope_nodes(path, {"Conv": "backbone_last_stage"})) == 1


def test_legacy_conv_scope_names_still_resolve(tmp_path):
    """Scope names from the previous profile ladder appear in published metadata."""
    path = tmp_path / "profiles.onnx"
    _save_profile_graph(path)
    assert _scope_nodes(path, {"Conv": "backbone_conv"}) == _scope_nodes(path, {"Conv": "backbone"})
    assert _scope_nodes(path, {"Conv": "all_conv"}) == _scope_nodes(path, {"Conv": "all"})


def test_true_and_false_are_scope_shorthands(tmp_path):
    path = tmp_path / "profiles.onnx"
    _save_profile_graph(path)
    assert _scope_nodes(path, {"Conv": True}) == _scope_nodes(path, {"Conv": "all"})
    assert _scope_nodes(path, {"Conv": "all", "Concat": False}) == _scope_nodes(path, {"Conv": "all"})


# A deliberately wide coverage setting. Spelled out rather than read from
# config so that narrowing the shipped settings -- which is the expected
# response to a failed accuracy gate -- does not turn into a test failure.
_WIDE_RTDETR_OPS = {
    "Conv": "all",
    "MaxPool": True,
    "AveragePool": True,
    "Resize": True,
    "Concat": True,
    "Add": "backbone",
    "Gemm": "ffn",
}


def test_each_op_type_scopes_independently(tmp_path):
    path = tmp_path / "profiles.onnx"
    _save_profile_graph(path)
    selected = _scope_nodes(path, dict(_WIDE_RTDETR_OPS))
    # 4 Conv, 1 MaxPool, 1 AveragePool, 1 Resize, 1 Concat, 1 Add, 2 Gemm
    assert len(selected) == 11
    # the Add scope is the backbone, so the decoder's Add is left alone
    assert "/model/backbone/res_layers.1/blocks.0/Add" in selected
    assert "/model/decoder/layers.0/Add" not in selected


def test_default_rtdetr_ops_exclude_geometry_and_head_gemms(tmp_path):
    """The shipped settings leave the numerically fragile transformer ops alone.

    sampling_offsets/attention_weights steer the deformable-attention
    GridSample, the score/bbox heads emit wide-range logits, and the raw
    MatMuls multiply two activations.
    """
    path = tmp_path / "profiles.onnx"
    _save_profile_graph(path)
    selected = _scope_nodes(path, dict(cfg.rtdetr_ptq_config["quantize_ops"]))
    for forbidden in ("sampling_offsets", "attention_weights", "_score_head", "_bbox_head"):
        assert not any(forbidden in name for name in selected), forbidden
    assert not any(name.endswith("/self_attn/MatMul") for name in selected)


def test_exclude_patterns_veto_any_scope(tmp_path):
    path = tmp_path / "profiles.onnx"
    _save_profile_graph(path)
    spec = _resolve_quantize_ops({"Conv": "all"}, RTDETR_SCOPES, ["/backbone/"])
    names, _ = _selected_nodes(str(path), spec)
    assert not any("/backbone/" in name for name in names)


def test_selected_nodes_reports_op_types(tmp_path):
    path = tmp_path / "profiles.onnx"
    _save_profile_graph(path)
    spec = _resolve_quantize_ops(dict(_WIDE_RTDETR_OPS), RTDETR_SCOPES)
    _, op_types = _selected_nodes(str(path), spec)
    assert set(op_types) == {"Conv", "MaxPool", "AveragePool", "Resize", "Concat", "Add", "Gemm"}


def test_decoder_scope_is_selectable(tmp_path):
    """Concat and Add document a "decoder" alternative, so it must resolve."""
    path = tmp_path / "profiles.onnx"
    _save_profile_graph(path)
    names = _scope_nodes(path, {"Add": "decoder"})
    assert names == ["/model/decoder/layers.0/Add"]


@pytest.mark.parametrize(
    ("arch_ops", "scopes"),
    [
        (cfg.rtdetr_ptq_config["quantize_ops"], RTDETR_SCOPES),
        (cfg.yolo_ptq_config["quantize_ops"], YOLO26_SCOPES),
    ],
)
def test_shipped_quantize_ops_name_scopes_the_architecture_has(arch_ops, scopes):
    """Every configured value must be True/False or a scope of that architecture.

    A scope name the architecture does not define would raise only at export
    time, after the FP32 baseline evaluation has already run.
    """
    for op_type, scope in arch_ops.items():
        if scope is True or scope is False:
            continue
        assert isinstance(scope, str), f"{op_type}: expected a scope name or True/False"
        assert _canonical_scope(scope) in scopes, f"{op_type}: unknown scope {scope!r}"


@pytest.mark.parametrize(
    ("arch_ops", "scopes"),
    [
        (cfg.rtdetr_ptq_config["quantize_ops"], RTDETR_SCOPES),
        (cfg.yolo_ptq_config["quantize_ops"], YOLO26_SCOPES),
    ],
)
def test_shipped_quantize_ops_resolve(arch_ops, scopes):
    """The shipped settings must survive _resolve_quantize_ops as they stand."""
    assert _resolve_quantize_ops(dict(arch_ops), scopes)


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError, match="Unsupported scope"):
        _resolve_quantize_ops({"Conv": "nonsense"}, RTDETR_SCOPES)


def test_all_ops_disabled_is_rejected():
    with pytest.raises(ValueError, match="no op types"):
        _resolve_quantize_ops({"Conv": False, "Gemm": False}, RTDETR_SCOPES)


def test_relaxation_hint_walks_the_order():
    # Starts from full coverage rather than the shipped settings, which have
    # already been narrowed and so begin part-way down the ladder.
    ops = {"Conv": "all", "Concat": True, "Add": "backbone_late", "Gemm": "ffn"}
    assert "'Gemm'] = False" in _relaxation_hint(ops)
    ops["Gemm"] = False
    assert "'Add'] = False" in _relaxation_hint(ops)
    ops["Add"] = False
    assert "'Concat'] = False" in _relaxation_hint(ops)
    ops["Concat"] = False
    assert '\'Conv\'] = "backbone_encoder"' in _relaxation_hint(ops)
    assert _relaxation_hint({"Conv": "backbone_late"}) == "Move to QAT"


# ── Per-architecture PTQ settings ────────────────────────────────────────────
def test_ptq_config_is_per_architecture():
    assert ptq_config("rtdetr") is cfg.rtdetr_ptq_config
    for arch in ("yolo", "yolo_ultra", "yolo_manual"):
        assert ptq_config(arch) is cfg.yolo_ptq_config


def test_architectures_have_independent_quantize_ops():
    """Changing one architecture's coverage must not affect the other."""
    assert cfg.yolo_ptq_config["quantize_ops"] is not cfg.rtdetr_ptq_config["quantize_ops"]
    assert cfg.yolo_ptq_config["exclude_patterns"] is not cfg.rtdetr_ptq_config["exclude_patterns"]


def test_yolo_scopes_do_not_accept_rtdetr_only_regions():
    """"ffn" names transformer Linears, which YOLO26 has none of."""
    assert "ffn" in RTDETR_SCOPES and "ffn" not in YOLO26_SCOPES
    with pytest.raises(ValueError, match="Unsupported scope"):
        _resolve_quantize_ops({"Gemm": "ffn"}, YOLO26_SCOPES)


# ── YOLO INT8 coverage profiles ────────────────────────────────────────────
def test_yolo_head_fp32_profile_uses_configured_exclusions():
    profile, patterns = _resolve_yolo_int8_profile("head_fp32")
    assert profile == "head_fp32"
    assert patterns == cfg.yolo_ptq_config["exclude_patterns"]


@pytest.mark.parametrize("requested", ["all_conv", "e2e"])
def test_yolo_full_int8_profile_removes_exclusions(requested):
    assert _resolve_yolo_int8_profile(requested) == ("all_conv", [])


def test_yolo_int8_profile_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported YOLO INT8 profile"):
        _resolve_yolo_int8_profile("unknown")


# ── Calibration ──────────────────────────────────────────────────────────────
def _make_reader(tmp_path, num_images=1, batch_size=1):
    paths = []
    for i in range(num_images):
        image_path = tmp_path / f"image{i}.jpg"
        cv2.imwrite(str(image_path), np.full((4, 5, 3), 127, dtype=np.uint8))
        paths.append(str(image_path))
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps({"images": [{"id": i, "file_name": p} for i, p in enumerate(paths)]})
    )

    def transform(*, image):
        return {"image": torch.from_numpy(image).permute(2, 0, 1).float() / 255.0}

    return CocoCalibrationDataReader(
        str(annotation_path),
        model_format="yolo",
        input_name="images",
        batch_size=batch_size,
        num_samples=num_images,
        transform=transform,
    )


def test_calibration_reader_returns_float32_and_rewinds(tmp_path):
    reader = _make_reader(tmp_path)
    first = reader.get_next()["images"]
    assert first.shape == (1, 3, 4, 5)
    assert first.dtype == np.float32
    assert first.flags.c_contiguous
    assert reader.get_next() is None

    reader.rewind()
    np.testing.assert_array_equal(first, reader.get_next()["images"])


def test_calibration_reader_drops_short_final_batch(tmp_path):
    """A static-batch graph cannot consume a partial batch."""
    reader = _make_reader(tmp_path, num_images=3, batch_size=2)
    assert reader.get_next()["images"].shape[0] == 2
    assert reader.get_next() is None


def test_chunked_reader_bounds_each_pass(tmp_path):
    """Histogram calibrators must be fed in bounded chunks (memory guard)."""
    chunked = _ChunkedCalibrationDataReader(_make_reader(tmp_path, num_images=5), chunk_size=2)

    served = []
    while chunked.start_chunk():
        count = 0
        while chunked.get_next() is not None:
            count += 1
        if count == 0:
            break
        served.append(count)
    assert served == [2, 2, 1]


# ── MSE calibration ──────────────────────────────────────────────────────────
def _mse_threshold(data, num_bins=2048):
    collector = _MSEHistogramCollector(symmetric=True, num_bins=num_bins)
    for chunk in data if isinstance(data, list) else [data]:
        collector.collect({"t": [np.asarray(chunk, dtype=np.float32)]})
    return float(collector.compute_collection_result()["t"].highest)


def test_mse_keeps_the_full_range_of_a_bounded_tensor():
    """Uniform data has no tail to trade away, so clipping can only hurt."""
    data = np.random.default_rng(0).uniform(-1.0, 1.0, 100_000).astype(np.float32)
    # Tolerance is a bin: the search only ever returns a histogram edge.
    threshold = _mse_threshold(data)
    assert threshold == pytest.approx(np.abs(data).max(), rel=1e-2)


def test_mse_clips_a_heavy_tail_without_cutting_the_bulk():
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 1.0, 200_000).astype(np.float32)
    data[:2_000] *= 50.0  # 1% of the mass, two orders of magnitude out

    threshold = _mse_threshold(data)
    assert threshold < np.abs(data).max()
    # The tail is what gets clipped, not the distribution it hangs off.
    assert threshold > 5.0


def test_mse_range_is_symmetric():
    """TensorRT requires symmetric activations with zero point 0."""
    collector = _MSEHistogramCollector(symmetric=True, num_bins=2048)
    data = np.random.default_rng(0).normal(3.0, 1.0, 50_000).astype(np.float32)
    collector.collect({"t": [data]})

    tensor_data = collector.compute_collection_result()["t"]
    assert float(tensor_data.lowest) == -float(tensor_data.highest)
    assert tensor_data.highest.dtype == np.float32


def test_mse_collects_across_chunks():
    """Chunked collection must see the whole set, not just the first chunk."""
    rng = np.random.default_rng(0)
    small = rng.normal(0.0, 1.0, 50_000).astype(np.float32)
    large = (rng.normal(0.0, 1.0, 50_000) * 20.0).astype(np.float32)

    chunked = _mse_threshold([small, large])
    at_once = _mse_threshold(np.concatenate([small, large]))
    assert chunked > _mse_threshold(small) * 5
    assert chunked == pytest.approx(at_once, rel=0.1)


def test_mse_rejects_asymmetric_ranges():
    with pytest.raises(ValueError, match="symmetric"):
        _MSEHistogramCollector(symmetric=False, num_bins=2048)


def test_mse_shares_the_percentile_cache_label():
    """MSE has no CalibrationMethod of its own; quantize_static checks the label."""
    assert _CALIBRATION_METHODS["mse"] == _CALIBRATION_METHODS["percentile"]


# ── YOLO end-to-end output decoding ──────────────────────────────────────────
def test_decode_yolo_e2e_rescales_to_original_image():
    import configs.export_cfg as cfg

    inp_h = cfg.input_height
    inp_w = cfg.input_width
    # One box covering the top-left quadrant of the network input.
    output0 = np.array([[[0.0, 0.0, inp_w / 2, inp_h / 2, 0.9, 3.0]]], dtype=np.float32)

    preds = decode_yolo_e2e_output(output0, [(200, 400, 7)], conf=0.1)
    assert len(preds) == 1
    assert preds[0]["image_id"] == 7
    assert preds[0]["category_id"] == 3
    assert preds[0]["bbox"] == pytest.approx([0.0, 0.0, 200.0, 100.0])


def test_decode_yolo_e2e_applies_confidence_threshold():
    output0 = np.array([[[0, 0, 10, 10, 0.9, 0], [0, 0, 10, 10, 0.01, 1]]], dtype=np.float32)
    assert len(decode_yolo_e2e_output(output0, [(100, 100, 1)], conf=0.5)) == 1


def test_decode_yolo_e2e_rejects_anchor_style_output():
    """A (batch, 4+nc, anchors) tensor must fail loudly, not score zero."""
    anchor_style = np.zeros((1, 14, 8400), dtype=np.float32)
    with pytest.raises(ValueError, match="end-to-end YOLO output"):
        decode_yolo_e2e_output(anchor_style, [(100, 100, 1)], conf=0.1)


def test_decode_yolo_e2e_ignores_padded_rows():
    """A static-batch graph is padded past the end of the dataset.

    The padded rows repeat a real image, so counting them would emit duplicate
    predictions against whichever image_id happened to follow.
    """
    detection = [0.0, 0.0, 10.0, 10.0, 0.9, 0.0]
    padded_batch = np.array([[detection], [detection]], dtype=np.float32)

    preds = decode_yolo_e2e_output(padded_batch, [(100, 100, 1)], conf=0.1)
    assert len(preds) == 1
    assert preds[0]["image_id"] == 1


# ── Evaluation dataset ───────────────────────────────────────────────────────
class _StubCoco:
    """Minimal stand-in for pycocotools' COCO index."""

    def __init__(self, images):
        self._images = {image["id"]: image for image in images}

    def loadImgs(self, image_id):
        return [self._images[image_id]]


def _passthrough_preprocess(*, image):
    return {"image": torch.from_numpy(image).permute(2, 0, 1)}


def test_eval_dataset_resolves_paths_against_the_annotation_file(tmp_path):
    """A relative file_name must resolve next to the annotations it came from."""
    cv2.imwrite(str(tmp_path / "frame.jpg"), np.full((8, 6, 3), 90, dtype=np.uint8))
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps({"images": []}))
    coco = _StubCoco([{"id": 5, "file_name": "frame.jpg"}])

    dataset = EvalDataset(
        coco, [5], None, _passthrough_preprocess, "yolo_manual", annotation_path=str(annotation_path)
    )
    image, orig_h, orig_w, image_id = dataset[0]
    assert (orig_h, orig_w, image_id) == (8, 6, 5)
    assert image.shape == (3, 8, 6)


def test_eval_dataset_raises_on_a_missing_image(tmp_path):
    """Dropping the image would leave its ground truth scored with no detections."""
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps({"images": []}))
    coco = _StubCoco([{"id": 5, "file_name": "gone.jpg"}])

    dataset = EvalDataset(
        coco, [5], None, _passthrough_preprocess, "yolo_manual", annotation_path=str(annotation_path)
    )
    with pytest.raises(FileNotFoundError, match="not found"):
        dataset[0]


def test_eval_dataset_raises_on_an_undecodable_image(tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    coco = _StubCoco([{"id": 5, "file_name": str(tmp_path / "broken.jpg")}])

    dataset = EvalDataset(coco, [5], None, _passthrough_preprocess, "yolo_manual")
    with pytest.raises(FileNotFoundError, match="could not be decoded"):
        dataset[0]


def test_eval_collate_keeps_every_image():
    """Silently shrinking a batch desynchronises predictions from the GT set."""
    batch = [(torch.zeros(3, 4, 4), 100, 200, i) for i in (1, 2, 3)]
    images, orig_hws = eval_collate_fn(batch)
    assert images.shape[0] == 3
    assert [hw[2] for hw in orig_hws] == [1, 2, 3]


# ── TensorRT precision contract ──────────────────────────────────────────────
def test_engine_rejects_qdq_graph_at_non_int8_precision():
    summary = {"explicit_qdq": True, "is_fp16": False}
    with pytest.raises(ValueError, match="Q/DQ nodes but precision"):
        _check_graph_matches_precision(summary, "fp16", "model.onnx")


def test_engine_rejects_int8_without_qdq():
    summary = {"explicit_qdq": False, "is_fp16": False}
    with pytest.raises(ValueError, match="no Q/DQ nodes"):
        _check_graph_matches_precision(summary, "int8", "model.onnx")


def test_engine_accepts_matching_fp32_graph():
    _check_graph_matches_precision({"explicit_qdq": False, "is_fp16": False}, "fp32", "model.onnx")


# ── Artifact naming ──────────────────────────────────────────────────────────
def test_artifact_paths_sit_next_to_the_weights():
    assert artifact_path("weights/yolo_manual_best.pth", "int8", ".onnx") == Path(
        "weights/yolo_manual_best_int8.onnx"
    )
    assert artifact_path("weights/rtdetr_best.pth", "fp16", ".engine") == Path(
        "weights/rtdetr_best_fp16.engine"
    )


def test_resolve_image_path_recovers_moved_data_path():
    existing = ROOT / "data" / "VisDrone" / "images" / "val" / "0000271_06401_d_0000404.jpg"
    stale = "/old/machine/project/data/VisDrone/images/val/0000271_06401_d_0000404.jpg"
    assert Path(resolve_image_path(stale)).resolve() == existing.resolve()


# ── Deployment graph profiling ───────────────────────────────────────────────
def _save_conv_graph(path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 3, 3])
    weight = numpy_helper.from_array(np.ones((4, 2, 1, 1), dtype=np.float32), "weight")
    bias = numpy_helper.from_array(np.ones((4,), dtype=np.float32), "bias")
    node = helper.make_node("Conv", ["x", "weight", "bias"], ["y"], name="conv")
    model = helper.make_model(
        helper.make_graph([node], "conv_profile", [x], [y], [weight, bias]),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.save(model, path)


def test_onnx_profiler_counts_parameters_and_macs(tmp_path):
    model_path = tmp_path / "model.onnx"
    _save_conv_graph(model_path)

    stats = profile_onnx_model(str(model_path))

    assert stats["params_M"] == pytest.approx(12 / 1e6)
    assert stats["macs_G"] == pytest.approx(72 / 1e9)
    assert stats["mac_nodes"] == 1
    assert stats["profile_basis"] == "deployed ONNX graph"


def test_trt_profiler_resolves_fp32_onnx_from_metadata(tmp_path):
    source = tmp_path / "model_fp32.onnx"
    engine = tmp_path / "model_int8.engine"
    _save_conv_graph(source)
    engine.write_bytes(b"test engine placeholder")
    Path(str(engine) + ".metadata.json").write_text(
        json.dumps({"fp32_onnx": str(source)})
    )

    assert resolve_profile_onnx(str(engine)) == source.resolve()
    stats = profile_deployment_artifact(str(engine))
    assert stats["params_M"] == pytest.approx(12 / 1e6)



def test_validation_sampling_zero_means_full_dataset(tmp_path, monkeypatch):
    annotation_path = tmp_path / "val.json"
    annotation_path.write_text(
        json.dumps({"images": [{"id": i, "file_name": f"{i}.jpg"} for i in range(10)]})
    )
    import export as export_module

    monkeypatch.setattr(export_module.cfg, "val_json", str(annotation_path))
    assert _validation_subsample_interval(0) == 1
    assert _validation_subsample_interval(3) == 4


@pytest.mark.parametrize("runtime", ["onnx", "trt"])
def test_export_evaluators_use_eval_config(runtime, monkeypatch):
    """Export settings do not define inference confidence or batch size."""
    import evaluate as evaluate_module
    import export as export_module

    calls = []
    monkeypatch.setitem(export_module.cfg_eval.eval_config, "conf_threshold", 0.123)
    monkeypatch.setitem(export_module.cfg_eval.eval_config, "batch_size", 4)
    monkeypatch.setattr(
        export_module, "_validation_subsample_interval", lambda *_args: 7
    )
    monkeypatch.setattr(
        export_module,
        "run_evaluation",
        lambda **kwargs: calls.append(kwargs) or {"mAP_50": 0.5},
    )

    if runtime == "onnx":
        session = object()
        monkeypatch.setattr(export_module.ort, "get_available_providers", lambda: [])
        monkeypatch.setattr(
            export_module,
            "_create_ort_session",
            lambda path, providers: session,
        )
        result = export_module.evaluate_onnx("model.onnx", "yolo", 100)
        assert calls[0]["model"] is session
        assert calls[0]["device"] == torch.device("cpu")
    else:
        wrapper = object()
        monkeypatch.setattr(evaluate_module, "TRTWrapper", lambda path: wrapper)
        result = export_module.evaluate_trt("model.engine", "yolo", 100)
        assert calls[0]["model"] is wrapper
        assert calls[0]["device"] == torch.device("cuda")

    assert result == {"mAP_50": 0.5}
    assert calls[0]["conf"] == pytest.approx(0.123)
    assert calls[0]["batch_size"] == 4
    assert calls[0]["subsample_interval"] == 7




def test_memory_benchmark_reports_process_rss_on_cpu():
    benchmark = _EvaluationMemoryBenchmark(torch.device("cpu"))
    benchmark.start()
    benchmark.sample()

    metrics = benchmark.metrics()
    assert metrics["process_rss_start_MB"] > 0
    assert metrics["process_rss_peak_MB"] >= metrics["process_rss_start_MB"]
    assert metrics["process_rss_delta_MB"] >= 0
    assert "cuda_device_memory_peak_MB" not in metrics


def test_evaluation_report_includes_memory_metrics(tmp_path):
    results = {
        "val/clean": {
            "mAP_50": 0.5,
            "mAP_50_95": 0.4,
            "mAP_75": 0.3,
            "AP_small": 0.2,
            "AP_medium": 0.3,
            "AP_large": 0.4,
            "AR_100": 0.5,
            "per_class_AP50": {},
            "latency_ms_mean": 10.0,
            "latency_ms_p95": 12.0,
            "fps": 100.0,
            "process_rss_peak_MB": 256.0,
            "process_rss_delta_MB": 8.0,
            "batch_size": 1,
            "subsample_interval": 1,
            "num_images": 2,
            "num_predictions": 3,
        }
    }

    generate_reports(results, "rtdetr", str(tmp_path))
    report = next(tmp_path.glob("eval_rtdetr_*.md")).read_text()

    assert "| Peak process RSS (MB) | 256.0 |" in report
    assert "| Peak CUDA device memory (MB) | N/A |" in report

def test_realtime_batch_size_is_enforced_for_static_artifacts():
    class FakeOnnx:
        @staticmethod
        def get_inputs():
            return [type("Input", (), {"shape": [1, 3, 736, 1280]})()]

    fake_trt = type("TRT", (), {"input_shape": (1, 3, 736, 1280)})()

    assert _resolve_evaluation_batch_size(object(), "rtdetr", 1) == 1
    assert _resolve_evaluation_batch_size(FakeOnnx(), "onnx", 1) == 1
    assert _resolve_evaluation_batch_size(fake_trt, "trt", 1) == 1

    with pytest.raises(ValueError, match="fixed batch 1"):
        _resolve_evaluation_batch_size(FakeOnnx(), "onnx", 4)
    with pytest.raises(ValueError, match="fixed batch 1"):
        _resolve_evaluation_batch_size(fake_trt, "trt", 4)




def test_failed_tensorrt_gate_preserves_last_published_engine(tmp_path, monkeypatch):
    import export as export_module

    weights = tmp_path / "model.pth"
    final_engine = export_module.artifact_path(str(weights), "int8", ".engine")
    final_metadata = Path(str(final_engine) + ".metadata.json")
    final_engine.write_bytes(b"known-good-engine")
    final_metadata.write_text('{"status": "known-good"}')

    built_paths = []

    def fake_build(source, engine_path, precision):
        built_paths.append(Path(engine_path))
        Path(engine_path).write_bytes(b"rejected-engine")
        Path(str(engine_path) + ".metadata.json").write_text(
            '{"status": "candidate"}'
        )
        return str(engine_path)

    monkeypatch.setattr(export_module, "build_engine_subprocess", fake_build)
    monkeypatch.setattr(
        export_module,
        "evaluate_trt",
        lambda *args, **kwargs: {
            "mAP_50": 0.0,
            "mAP_50_95": 0.0,
            "subsample_interval": 1,
        },
    )

    with pytest.raises(RuntimeError, match="accuracy gate failed"):
        export_module.export_engine(
            str(weights),
            "rtdetr",
            "int8",
            {
                "model": "candidate.onnx",
                "fp32_onnx": "baseline.onnx",
                "fp32_metrics": {"mAP_50": 0.8, "mAP_50_95": 0.6},
            },
        )

    assert len(built_paths) == 1
    assert built_paths[0] != final_engine
    assert built_paths[0].parent.name.startswith(".export_")
    assert final_engine.read_bytes() == b"known-good-engine"
    assert final_metadata.read_text() == '{"status": "known-good"}'


def test_calibration_pool_does_not_follow_the_annotation_budget():
    """A calibrator collects activation ranges and never opens an annotation.

    Following the budget would draw each arm's calibration images from a
    different pool, so two INT8 engines differing only in training budget would
    also differ in what they were calibrated on.
    """
    import importlib

    import configs.base_cfg as base_cfg
    import configs.export_cfg as export_cfg

    original = base_cfg.label_budget
    pools = set()
    try:
        for budget in (None, *sorted(base_cfg.label_budget_splits)):
            base_cfg.label_budget = budget
            importlib.reload(export_cfg)
            pools.add(export_cfg.yolo_ptq_config["calibration_annotations"])
            pools.add(export_cfg.rtdetr_ptq_config["calibration_annotations"])
    finally:
        base_cfg.label_budget = original
        importlib.reload(export_cfg)

    assert len(pools) == 1, f"calibration pool varied by budget: {pools}"
    assert pools.pop().endswith("train.json")


# --- the accuracy gate scores on the set the checkpoint was selected on -------

def _checkpoint_with_validation(tmp_path, manifest, targets):
    import torch
    from experiment_artifacts import RUN_PROVENANCE_KEY

    path = tmp_path / "student.pth"
    torch.save({
        "model": {},
        RUN_PROVENANCE_KEY: {
            "version": 1,
            "identity": {"validation": {"manifest": str(manifest), "targets": targets}},
        },
    }, path)
    return str(path)


def test_gate_uses_the_manifest_the_run_was_selected_against(tmp_path):
    """A teacher-only student was selected on the full pool with teacher targets.

    Gating it on budgeted ground truth would contradict its own selection set and
    read labels its scenario says were never bought.
    """
    import export

    manifest = tmp_path / "val_teacher_pseudo.json"
    manifest.write_text('{"images": [{"id": 1}], "annotations": [], "categories": []}')
    checkpoint = _checkpoint_with_validation(tmp_path, manifest, "teacher")

    assert export.gate_validation_manifest(checkpoint) == (str(manifest), "teacher", "")


def test_gate_falls_back_to_the_budgeted_split_for_a_supervised_checkpoint(tmp_path):
    import export
    import configs.export_cfg as export_cfg

    manifest = tmp_path / "val_half.json"
    manifest.write_text('{"images": [{"id": 1}], "annotations": [], "categories": []}')
    checkpoint = _checkpoint_with_validation(tmp_path, manifest, "ground_truth")
    assert export.gate_validation_manifest(checkpoint) == (str(manifest), "ground_truth", "")

    # Nothing recorded at all -- every legacy artifact.
    import torch
    legacy = tmp_path / "legacy.pth"
    torch.save({"model": {}}, legacy)
    manifest, targets, reason = export.gate_validation_manifest(str(legacy))
    assert (manifest, targets) == (export_cfg.val_json, "ground_truth")
    assert "provenance" in reason


def test_gate_falls_back_when_the_recorded_manifest_is_gone(tmp_path):
    """Teacher-scored target files are content-addressed and may be pruned."""
    import export
    import configs.export_cfg as export_cfg

    checkpoint = _checkpoint_with_validation(tmp_path, tmp_path / "absent.json", "teacher")
    manifest, targets, reason = export.gate_validation_manifest(checkpoint)
    assert (manifest, targets) == (export_cfg.val_json, "ground_truth")
    # The substitution changes what is measured, so it must name itself.
    assert "missing" in reason and "absent.json" in reason and "teacher" in reason


@pytest.mark.parametrize("weights", [None, "", "/does/not/exist.pth"])
def test_gate_falls_back_without_a_readable_checkpoint(weights):
    import export
    import configs.export_cfg as export_cfg

    manifest, targets, reason = export.gate_validation_manifest(weights)
    assert (manifest, targets) == (export_cfg.val_json, "ground_truth")
    assert reason


def test_subsample_interval_is_computed_on_the_gate_manifest(tmp_path):
    """Otherwise the interval is derived from a different set than the one scored."""
    import export

    manifest = tmp_path / "pool.json"
    manifest.write_text(json.dumps({"images": [{"id": i} for i in range(1, 201)]}))
    assert export._validation_subsample_interval(100, str(manifest)) == 2
    assert export._validation_subsample_interval(0, str(manifest)) == 1


def test_a_recorded_absolute_path_is_retried_under_the_current_project_root(tmp_path, monkeypatch):
    """Checkpoints record absolute paths; a moved or cloned checkout must still resolve."""
    import export

    monkeypatch.setattr(export, "PROJECT_ROOT", str(tmp_path))
    annotations = tmp_path / "data" / "processed_annotations"
    annotations.mkdir(parents=True)
    real = annotations / "val_teacher_pseudo_half.json"
    real.write_text(json.dumps({"images": [{"id": 1}], "annotations": [], "categories": []}))

    stale = "/somewhere/else/data/processed_annotations/val_teacher_pseudo_half.json"
    checkpoint = _checkpoint_with_validation(tmp_path, stale, "teacher")

    manifest, targets, reason = export.gate_validation_manifest(checkpoint)
    assert manifest == str(real) and targets == "teacher" and reason == ""
