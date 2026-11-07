"""The single seed-derivation function (contracts §0, RNG row).

Every random draw in Sevi uses ``numpy.random.Generator(numpy.random.PCG64(seed))`` with a seed
from :func:`derive_seed`. No module may call ``random``, an unseeded ``np.random.default_rng()``
or the clock for randomness.
"""

import hashlib

SEED_DOMAIN = "sevi-seed-v1"


def derive_seed(scope_id: str, purpose: str) -> int:
    """First 8 bytes (big-endian) of ``sha256(f"sevi-seed-v1|{scope_id}|{purpose}")``.

    The separators are literal pipe characters; the result is a UInt64.
    """
    digest = hashlib.sha256(f"{SEED_DOMAIN}|{scope_id}|{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
