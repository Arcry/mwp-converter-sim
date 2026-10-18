"""Сверка API OptiCommPy (шаг 3.1): печатает сигнатуры и docstring нужных функций."""
import importlib
import inspect

FUNCS = [
    ("optic.models.devices", "mzm"),
    ("optic.models.devices", "photodiode"),
    ("optic.models.devices", "basicLaserModel"),
    ("optic.comm.modulation", "modulateGray"),
    ("optic.comm.modulation", "demodulateGray"),
    ("optic.comm.modulation", "grayMapping"),
    ("optic.comm.metrics", "calcEVM"),
    ("optic.comm.metrics", "fastBERcalc"),
    ("optic.comm.metrics", "theoryBER"),
    ("optic.dsp.core", "pulseShape"),
    ("optic.dsp.core", "firFilter"),
    ("optic.dsp.core", "pnorm"),
    ("optic.utils", "parameters"),
]


def main(with_doc=True):
    for mod_name, name in FUNCS:
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, name)
        try:
            sig = str(inspect.signature(obj))
        except (TypeError, ValueError):
            sig = "(class/module)"
        print(f"=== {mod_name}.{name}{sig}")
        if with_doc:
            print(inspect.getdoc(obj))
        print()


if __name__ == "__main__":
    main()
