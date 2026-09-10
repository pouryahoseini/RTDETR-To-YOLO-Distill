"""Build TensorRT engines from ONNX graphs.

Run this module on the target NVIDIA system: an engine plan is tied to the
TensorRT version and the GPU it was built for.

Precision policy
----------------
TensorRT 11 is *strongly typed*.  ``BuilderFlag.FP16``, ``BuilderFlag.INT8``,
``IBuilderConfig.int8_calibrator`` and the whole ``IInt8Calibrator`` family were
removed, and every network is created with ``STRONGLY_TYPED`` whether or not the
flag is passed.  An engine therefore executes exactly the types present in the
ONNX graph, and this builder refuses to pretend otherwise:

  * ``fp32``  - parse an FP32 graph as-is.
  * ``fp16``  - the graph must already be FP16 (``export.convert_onnx_to_fp16``).
  * ``int8``  - the graph must carry explicit Q/DQ nodes with the scales baked
                in (``export.quantize_onnx_int8``).  No calibration happens here.

On TensorRT 8/10 the builder flags still exist, so FP32/FP16 graphs are accepted
for every precision and the corresponding flags are set instead.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import configs.export_cfg as cfg

try:
    import tensorrt as trt
except ImportError:  # pragma: no cover - environment guard
    trt = None

PRECISIONS = ("fp32", "fp16", "int8")


def trt_major_version() -> int:
    """Return the installed TensorRT major version, or 0 when absent."""
    if trt is None:
        return 0
    try:
        return int(str(trt.__version__).split(".", 1)[0])
    except (TypeError, ValueError):
        return 0


def describe_onnx(onnx_path: str) -> dict:
    """Summarize the precision-relevant properties of an ONNX graph."""
    import onnx
    from onnx import TensorProto

    model = onnx.load(onnx_path, load_external_data=False)
    quantize_nodes = sum(n.op_type == "QuantizeLinear" for n in model.graph.node)
    dequantize_nodes = sum(n.op_type == "DequantizeLinear" for n in model.graph.node)
    fp16_elems = sum(
        int(np.prod(init.dims)) for init in model.graph.initializer if init.data_type == TensorProto.FLOAT16
    )
    fp32_elems = sum(
        int(np.prod(init.dims)) for init in model.graph.initializer if init.data_type == TensorProto.FLOAT
    )
    return {
        "quantize_nodes": quantize_nodes,
        "dequantize_nodes": dequantize_nodes,
        "explicit_qdq": quantize_nodes > 0,
        "is_fp16": fp16_elems > fp32_elems,
    }


def _check_graph_matches_precision(summary: dict, precision: str, onnx_path: str):
    """Reject graph/precision combinations that would silently build FP32."""
    if summary["explicit_qdq"] and precision != "int8":
        raise ValueError(
            f"{onnx_path} carries Q/DQ nodes but precision '{precision}' was requested. "
            "Build it with --precision int8 or export a non-quantized graph."
        )
    if precision == "int8" and not summary["explicit_qdq"]:
        if trt_major_version() >= 11:
            raise ValueError(
                f"{onnx_path} has no Q/DQ nodes. TensorRT {trt.__version__} removed the INT8 "
                "calibrator, so INT8 requires an explicit-QDQ graph. Produce one with "
                "`python src/export.py ... --precision int8`."
            )
        raise ValueError(
            f"{onnx_path} has no Q/DQ nodes. Implicit calibration inside the engine builder is "
            "not supported by this project; export an explicit-QDQ INT8 ONNX instead."
        )
    if precision == "fp32" and summary["is_fp16"]:
        raise ValueError(
            f"{onnx_path} is an FP16 graph but precision 'fp32' was requested; the engine would "
            "silently run in half precision. Use --precision fp16 or export an FP32 graph."
        )
    if precision == "fp16" and trt_major_version() >= 11 and not summary["is_fp16"]:
        raise ValueError(
            f"{onnx_path} is an FP32 graph. TensorRT {trt.__version__} is strongly typed and has no "
            "FP16 builder flag, so an FP16 engine needs an FP16 graph. Produce one with "
            "`python src/export.py ... --precision fp16`."
        )


def _configure_precision_flags(config, precision: str):
    """Set the legacy builder flags used by TensorRT 8/10."""
    if trt_major_version() >= 11:
        return  # Strongly typed: the graph alone determines precision.
    if precision == "int8" and hasattr(trt.BuilderFlag, "INT8"):
        # NVIDIA requires the INT8 flag even for explicit-QDQ networks on TRT<=10
        # so INT8 tactics are available; FP16 covers the layers left unquantized.
        config.set_flag(trt.BuilderFlag.INT8)
    if precision in ("fp16", "int8") and hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)


def build_engine(onnx_path: str, engine_path: str, precision: str = "fp32") -> str:
    """Build and serialize a TensorRT engine, then write a provenance sidecar."""
    if trt is None:
        raise ImportError("TensorRT must be installed on the target NVIDIA system.")
    if precision not in PRECISIONS:
        raise ValueError(f"Unsupported precision: {precision}")

    summary = describe_onnx(onnx_path)
    _check_graph_matches_precision(summary, precision, onnx_path)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # EXPLICIT_BATCH is the only mode from TensorRT 10 onwards and the flag is gone.
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, logger)
    onnx_bytes = Path(onnx_path).read_bytes()
    if not parser.parse(onnx_bytes):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    workspace = int(float(cfg.export_config["workspace_gb"]) * 1024**3)
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    else:
        config.max_workspace_size = workspace
    if hasattr(config, "profiling_verbosity") and hasattr(trt, "ProfilingVerbosity"):
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    _configure_precision_flags(config, precision)

    shapes = [tuple(network.get_input(i).shape) for i in range(network.num_inputs)]
    if any(dim < 0 for shape in shapes for dim in shape):
        raise ValueError(
            f"{onnx_path} has dynamic input shapes {shapes}. Export with "
            "export_config['dynamic_axes'] = False for edge deployment."
        )

    print(f"Building {precision.upper()} engine from {onnx_path} (inputs {shapes})...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the serialized engine.")

    engine = Path(engine_path)
    engine.parent.mkdir(parents=True, exist_ok=True)
    staged = engine.with_suffix(engine.suffix + ".tmp")
    staged.write_bytes(serialized)
    os.replace(staged, engine)

    with open(str(engine) + ".metadata.json", "w") as f:
        json.dump(
            {
                "engine": str(engine.resolve()),
                "onnx": str(Path(onnx_path).resolve()),
                "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
                "tensorrt_version": str(trt.__version__),
                "strongly_typed": trt_major_version() >= 11,
                "precision": precision,
                **summary,
            },
            f,
            indent=2,
        )
    print(f"Saved TensorRT engine to {engine}")
    return str(engine)


def main():
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX graph.")
    parser.add_argument("--onnx", required=True, help="Path to the input ONNX model.")
    parser.add_argument("--engine", help="Output engine path (defaults next to the ONNX).")
    parser.add_argument("--precision", default="fp32", choices=PRECISIONS)
    args = parser.parse_args()

    engine_path = args.engine or str(Path(args.onnx).with_suffix(".engine"))
    build_engine(args.onnx, engine_path, precision=args.precision)


if __name__ == "__main__":
    main()
