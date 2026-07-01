from dataclasses import dataclass, field
from pathlib import Path

import yaml

from mxsep.utils.serialize import asdict_filter_empty

path = "kaggle/store.yaml"

@dataclass
class KaggleStore:
    
    dataset_env:str = field(default_factory=dict)

    user_api_keys:dict[str,str] = field(default_factory=dict)

    dataset_paths:dict[str,str] = field(default_factory=dict)

    preprocessed_dataset_paths:dict[str,str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> 'KaggleStore':
        # Create a copy to avoid mutating the input
        data_copy = data.copy()
        return cls(**data_copy)
    
    def to_dict(self) -> dict:
        return asdict_filter_empty(self)

    def save_store(self)->None:
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f)


    @classmethod
    def load_store(cls)->'KaggleStore':
        if Path(path).exists():
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
    
            return KaggleStore.from_dict(data)
        else:
            return cls.init_store()
    
    @classmethod
    def init_store(cls)->'KaggleStore':
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        store = cls()
        store.save_store()
        return store