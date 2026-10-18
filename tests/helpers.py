from dataclasses import replace

from mwp_converter.params import load_config, quick

import conftest

CFG = conftest.ROOT / "configs"


def cfg_default(**ov):
    return load_config(CFG / "default.toml", ov)


def cfg_quick(**ov):
    return quick(cfg_default(**ov))


__all__ = ["replace", "CFG", "cfg_default", "cfg_quick"]
