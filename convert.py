#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust ZeRO->fp32 converter for Torch>=2.6 (weights_only=True default).
It (1) pre-allowlists common DeepSpeed symbols; (2) on failure, parses the
'Unsupported global: GLOBAL ...' from the exception, allowlists it, and retries.
"""

import argparse
import re
import importlib
from pathlib import Path

def _has_add_safe_globals():
    try:
        from torch.serialization import add_safe_globals  # noqa: F401
        return True
    except Exception:
        return False

def _add_safe(objs):
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals(objs)
    except Exception:
        pass

def _try_import_symbol(qualname: str):
    """
    Import 'a.b.c' -> returns object 'c' from module 'a.b'.
    Returns None if anything fails.
    """
    try:
        mod_name, attr = qualname.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    except Exception:
        return None

def _pre_allowlist_commons():
    # Pre-allowlist common DS symbols seen in ZeRO shards
    commons = [
        # FP16 scalers
        "deepspeed.runtime.fp16.loss_scaler.LossScaler",
        "deepspeed.runtime.fp16.dynamic_loss_scaler.DynamicLossScaler",
        # ZeRO enums/config/status
        "deepspeed.runtime.zero.config.ZeroStageEnum",
        "deepspeed.runtime.zero.stage_1_and_2.ZeroParamStatus",
        "deepspeed.runtime.zero.stage_1_and_2.ZeroOptimizerStage2",
        "deepspeed.runtime.config.DeepSpeedConfig",
        # You just hit this one:
        "deepspeed.utils.tensor_fragment.fragment_address",
    ]
    objs = []
    for qn in commons:
        obj = _try_import_symbol(qn)
        if obj is not None:
            objs.append(obj)
    if objs:
        _add_safe(objs)

def _extract_unsupported_globals(msg: str):
    """
    Parse error text for lines like:
    'Unsupported global: GLOBAL deepspeed.utils.tensor_fragment.fragment_address'
    Return list of qualified names.
    """
    pats = [
        r"Unsupported global:\s+GLOBAL\s+([A-Za-z0-9_\.]+)",
        r"was not an allowed global.*?\[\s*([A-Za-z0-9_\.]+)\s*\]",
    ]
    found = set()
    for pat in pats:
        for m in re.finditer(pat, msg):
            found.add(m.group(1))
    return list(found)

def convert_zero_to_fp32(ckpt_dir: str, out_path: str, max_retries: int = 5):
    from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

    # Step 0: pre-allowlist common DS symbols (no-op on old torch)
    if _has_add_safe_globals():
        _pre_allowlist_commons()

    # Step 1: try convert; on failure, parse & allowlist missing globals, then retry
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            convert_zero_checkpoint_to_fp32_state_dict(ckpt_dir, out_path)
            print(f"[OK] Converted ZeRO checkpoint → {out_path}")
            return
        except Exception as e:
            last_err = e
            msg = str(e)
            missing = _extract_unsupported_globals(msg) if _has_add_safe_globals() else []
            if not missing:
                # nothing to auto-allowlist or on old torch -> just bail
                break
            objs = []
            for qn in missing:
                obj = _try_import_symbol(qn)
                if obj is not None:
                    objs.append(obj)
            if objs:
                _add_safe(objs)
                print(f"[Retry {attempt}/{max_retries}] allowlisted: {', '.join(missing)}; retrying…")
                continue
            else:
                # couldn't import any of them
                break
    # If we reach here, conversion failed
    raise last_err

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  type=str, required=True,
                        help='Path to the ZeRO checkpoint folder (…/epoch=XX.ckpt/checkpoint)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output fp32 PyTorch state_dict file')
    args = parser.parse_args()

    ckpt_dir = Path(args.input)
    out = Path(args.output) if args.output is not None else (ckpt_dir / 'converted.ckpt')

    convert_zero_to_fp32(str(ckpt_dir), str(out))

if __name__ == '__main__':
    main()



# import argparse
# from pathlib import Path
# from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

# if __name__ == '__main__':
#     ## read a path using argparse and pass it to convert_zero_checkpoint_to_fp32_state_dict
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input', type=str, default=None, help='path to the desired checkpoint folder')
#     parser.add_argument('--output', type=str, default=None, help='path to the pytorch fp32 state_dict output file')
#     # parser.add_argument('--tag', type=str, help='checkpoint tag used as a unique identifier for checkpoint')
#     args = parser.parse_args()
#     if args.output is None:
#         args.output = Path(args.input) / 'converted.ckpt'
#     convert_zero_checkpoint_to_fp32_state_dict(args.input, args.output)