import sys
import yaml

class GSNet:
    def __init__(self) -> None:
        self._read_params()

    def _read_params(self):
        cfg_path = sys.path[0]+'/config/gsnet.yaml'
        with open(cfg_path, 'r') as config:
            self.cfg = yaml.safe_load(config)