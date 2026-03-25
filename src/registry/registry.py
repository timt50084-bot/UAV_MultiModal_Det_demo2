from collections.abc import Mapping

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover
    OmegaConf = None


class Registry:
    """Lightweight registry for config-driven module construction."""

    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register(self, module_name=None):
        def _register(cls):
            name = module_name if module_name is not None else cls.__name__
            if name in self._module_dict:
                raise KeyError(f"Module {name} is already registered in {self._name}.")
            self._module_dict[name] = cls
            return cls

        return _register

    def _to_dict(self, cfg):
        if OmegaConf is not None and OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)

        if not isinstance(cfg, Mapping):
            raise TypeError(f"cfg must be a mapping, but got {type(cfg)}")

        return dict(cfg)

    def build(self, cfg, **kwargs):
        cfg_dict = self._to_dict(cfg)
        if 'type' not in cfg_dict:
            raise KeyError(f"cfg must contain a 'type' key, got: {cfg_dict}")

        obj_type = cfg_dict['type']
        if obj_type not in self._module_dict:
            raise KeyError(
                f"'{obj_type}' is not registered in {self._name}. "
                f"Available: {list(self._module_dict.keys())}"
            )

        obj_cls = self._module_dict[obj_type]
        args = {k: v for k, v in cfg_dict.items() if k != 'type'}
        args.update(kwargs)
        return obj_cls(**args)
