from dataclasses import asdict
from enum import Enum


def asdict_filter_empty(obj):
    """Convert dataclass to dict, filtering out empty values."""

    def is_empty(value):
        return value is None or value == {} or value == []

    def filter_dict(d):
        result = {}
        for k, v in d.items():
            if is_empty(v):
                continue
            if isinstance(v, Enum):
                result[k] = v.value
            elif isinstance(v, tuple):
                result[k] = list(v)
            elif isinstance(v, dict):
                filtered = filter_dict(v)
                if filtered:  # Only add non-empty dicts
                    result[k] = filtered
            elif isinstance(v, list):
                filtered_list = []
                for item in v:
                    if isinstance(item, dict):
                        filtered_item = filter_dict(item)
                        if filtered_item:
                            filtered_list.append(filtered_item)
                    elif isinstance(item, list):
                        # Handle nested lists if needed
                        filtered_list.append(item)
                    elif not is_empty(item):
                        filtered_list.append(item)
                if filtered_list:
                    result[k] = filtered_list
            else:
                result[k] = v
        return result

    d = asdict(obj)
    return filter_dict(d)
