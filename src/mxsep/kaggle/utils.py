import hashlib
from typing import Any, Dict

import yaml


def _dict_to_hash(config_dict: Dict[str, Any]) -> str:
    """
    Create a hash from a config dictionary.

    Args:
        config_dict: Configuration dictionary (originally from YAML)

    Returns:
        SHA-256 hash as a hexadecimal string
    """
    # Convert dict to sorted JSON string for consistent hashing
    # Sort keys to ensure the same dict always produces the same hash
    import json
    json_str = json.dumps(config_dict, sort_keys=True, default=str)

    # Create SHA-256 hash
    hash_obj = hashlib.sha256(json_str.encode('utf-8'))
    return hash_obj.hexdigest()


def _hash_to_short_id(hash_str: str, length: int = 8) -> str:
    """
    Convert a hex hash to a short ID using base36 encoding.

    Args:
        hash_str: Hexadecimal hash string
        length: Desired length of the short ID (default: 8)

    Returns:
        Short ID using characters 0-9a-z
    """
    # Allowed characters: 0-9a-z
    allowed_chars = '0123456789abcdefghijklmnopqrstuvwxyz'
    base = len(allowed_chars)

    # Convert hex string to integer
    hash_int = int(hash_str, 16)

    # Convert to base36 (0-9a-z)
    short_id = []
    while hash_int > 0:
        remainder = hash_int % base
        short_id.append(allowed_chars[remainder])
        hash_int //= base

    # Pad with leading zeros if needed
    while len(short_id) < length:
        short_id.append('0')

    # Reverse to get correct order
    short_id.reverse()

    # If the ID is longer than desired length
    if len(short_id) > length:
        # least significant digits for better distribution
        short_id = short_id[-length:]

    return ''.join(short_id)


def create_id_from_config(config_dict: Dict[str, Any], id_length: int = 8) -> str:
    """
    Create a short ID from a config dictionary in one step.

    Args:
        config_dict: Configuration dictionary
        id_length: Desired ID length (default: 8)

    Returns:
        8-character ID using characters 0-9a-z
    """
    hash_str = _dict_to_hash(config_dict)
    return _hash_to_short_id(hash_str, id_length)
