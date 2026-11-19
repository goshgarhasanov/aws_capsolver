import base64
import hashlib

import pytest

from aws_capsolver.waf import pow as powlib


def _reference(kind, base, salt, bits, memory=128):
    """Independent, bit-string based reference search."""
    for i in range(10_000_000):
        data = f"{base}{i}".encode()
        if kind == "sha256":
            d = hashlib.sha256(data).digest()
        else:
            d = hashlib.scrypt(data, salt=salt.encode(), n=memory, r=8, p=1, dklen=16)
        if "".join(f"{b:08b}" for b in d).startswith("0" * bits):
            return str(i)


@pytest.mark.parametrize("digest,bits,ok", [
    (bytes([0x00, 0x0F]), 12, True),
    (bytes([0x00, 0x1F]), 12, False),
    (bytes([0x00, 0x00, 0x80]), 16, True),
    (bytes([0x80]), 0, True),
    (bytes([0x7F]), 1, True),
    (bytes([0x7F]), 2, False),
])
def test_leading_zero_bits(digest, bits, ok):
    assert powlib.leading_zero_bits_ok(digest, bits) is ok


def test_sha256_known_vector():
    # sha256("abc" + "hm" + "39") style vectors: compare against the reference search
    expected = _reference("sha256", "inputhmac", "hmac", 12)
    assert powlib.solve_hashcash("sha256", "input", "hmac", 12, processes=1) == expected
    digest = hashlib.sha256(f"inputhmac{expected}".encode()).digest()
    assert digest[0] == 0 and digest[1] >> 4 == 0


def test_sha256_parallel_matches_sequential():
    seq = powlib.solve_hashcash("sha256", "challenge-x", "", 17, processes=1)
    par = powlib.solve_hashcash("sha256", "challenge-x", "", 17, processes=4)
    assert seq == par == _reference("sha256", "challenge-x", "", 17)


def test_scrypt_vector_and_parallel():
    expected = _reference("scrypt", "in" + "salt", "salt", 6)
    assert powlib.solve_hashcash("scrypt", "in", "salt", 6, processes=1) == expected
    assert powlib.solve_hashcash("scrypt", "in", "salt", 9, processes=4) == \
        _reference("scrypt", "insalt", "salt", 9)


def test_hashcash_budget_exhausted():
    assert powlib.solve_hashcash("sha256", "x", "", 40, processes=1, max_iterations=50) is None


def test_bandwidth_sizes():
    assert len(base64.b64decode(powlib.solve_bandwidth(1))) == 1024
    assert len(base64.b64decode(powlib.solve_bandwidth(3))) == 0x19000
    with pytest.raises(ValueError):
        powlib.solve_bandwidth(9)


def test_bad_difficulty():
    with pytest.raises(ValueError):
        powlib.solve_hashcash("sha256", "x", "", -1)


def test_hashcash_deadline():
    import time
    assert powlib.solve_hashcash("sha256", "x", "", 60, processes=2,
                                 deadline=time.monotonic() - 1) is None
