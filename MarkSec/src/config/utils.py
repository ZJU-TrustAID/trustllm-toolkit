from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, ValidationError
from src.exceptions.exceptions import ExpectedStringPathError


def load_yaml(cfg_path: str | List[str]) -> Dict:

    if isinstance(cfg_path, list):
        raise ExpectedStringPathError(cfg_path)

    with open(cfg_path, "r") as stream:
        try:
            yaml_obj = yaml.safe_load(stream)
            return yaml_obj
        except (yaml.YAMLError, ValidationError) as exc:
            print(exc)
            raise exc


class PydanticBaseModelWithOptionalDefaultsPath(BaseModel):
    def __init__(self, **kwargs: Any) -> None:
        if "cfg_path" in kwargs:
            cfg = load_yaml(kwargs["cfg_path"])
            del kwargs["cfg_path"]
            cfg |= kwargs
        else:
            cfg = kwargs
        super().__init__(**cfg)