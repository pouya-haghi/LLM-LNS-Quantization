# LLM-LNS-Quantization

This repository updates LMEvalHarness to add new quantization methods to it.

The new formats are:
- LNS8, LNS4
- Dynamic LNS with optimizations (per-block quantization) in 8 and 4 bits
- FP8, FP4
- MX block floating-point
- ZeroQuant
- VSQuant
- INT8 (W8A8)
- SmoothQuant
- LLM.int8()

The file is under `lm_eval/models/huggingface.py`.
