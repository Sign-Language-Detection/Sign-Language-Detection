import os
import yaml
import importlib


class MLModelLoader:
    """
    Dynamically loads and initializes models defined in a YAML config.
    """

    def __init__(self, config_path: str):
        # 1) Load the YAML
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.models_cfg = cfg.get("models", {})
        # Cache of instantiated models
        self._instances = {}
        # Keep the path for resolving relative model_path entries
        self._config_dir = os.path.dirname(config_path)

    def get(self, key: str):
        """
        Retrieve the model instance for the given key (e.g. "sign_detect").
        If it’s not yet instantiated, import, initialize, cache, and return it.
        """
        # Return cached if already loaded
        if key in self._instances:
            return self._instances[key]

        # Make sure the key exists in the YAML under `models:`
        if key not in self.models_cfg:
            raise KeyError(f"Model '{key}' not found in config.")

        mconf = self.models_cfg[key]
        module_path = mconf.get("module")
        class_name  = mconf.get("class")
        model_path  = mconf.get("model_path")

        # module & class are required fields
        if not module_path or not class_name:
            raise ValueError(f"Model '{key}' must specify both 'module' and 'class'.")

        # Resolve relative model_path against the config file directory
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(self._config_dir, model_path)

        # Dynamically import the module and get the class
        module = importlib.import_module(module_path)
        cls    = getattr(module, class_name)

        # Instantiate, passing model_path if the constructor accepts it
        try:
            instance = cls(model_path=model_path)
        except TypeError:
            # Fallback if the constructor doesn’t accept model_path
            instance = cls()

        for hook in ("load_models", "load_model", "init_model"):
            if hasattr(instance, hook):
                getattr(instance, hook)()
                break


        # Cache and return
        self._instances[key] = instance
        return instance

    def model_processor(self, key: str, method: str, *args, **kwargs):
        """
        Retrieve the model for 'key' and call its 'method' with any args/kwargs.
        Example:
          loader.model_processor("sign_detect", "predict_frame", frame)
          loader.model_processor("sign_detect", "run_realtime")
        """
        model = self.get(key)
        if not hasattr(model, method):
            raise AttributeError(f"Model '{key}' has no method '{method}'.")
        func = getattr(model, method)
        return func(*args, **kwargs)

