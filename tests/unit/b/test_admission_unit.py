"""Admission pure helpers (B spec §9.1, §9.2; contracts §9.2.1, §9.3)."""

from __future__ import annotations

import ipaddress

import pytest

from sevi.app.admission import client_ip, client_net, eta_s, ip_hmac, is_public_ip, ws_url

TRUSTED = [ipaddress.ip_network("127.0.0.1/32"), ipaddress.ip_network("::1/128")]
XFF = "X-Forwarded-For"


def test_U_ING_01_rightmost_xff_from_trusted_peer() -> None:
    h = {"x-forwarded-for": "1.2.3.4, 203.0.113.9"}
    assert str(client_ip("127.0.0.1", h, TRUSTED, XFF)) == "203.0.113.9"


@pytest.mark.parametrize("right", ["10.0.0.7", "100.64.1.2", "192.168.1.1", "127.0.0.1", "fe80::1"])
def test_U_ING_01_non_public_rightmost_is_none(right: str) -> None:
    assert client_ip("127.0.0.1", {"x-forwarded-for": f"8.8.8.8, {right}"}, TRUSTED, XFF) is None


def test_U_ING_01_empty_header_key_is_none() -> None:
    assert client_ip("127.0.0.1", {"x-forwarded-for": "203.0.113.9"}, TRUSTED, "") is None


def test_IT_API_02_cf_connecting_ip_and_untrusted_peer() -> None:
    cf = {"cf-connecting-ip": "203.0.113.9"}
    assert str(client_ip("127.0.0.1", cf, TRUSTED, "CF-Connecting-IP")) == "203.0.113.9"
    # untrusted peer: every client-IP header is ignored
    assert str(client_ip("10.0.0.5", {"x-forwarded-for": "203.0.113.9"}, TRUSTED, XFF)) == (
        "10.0.0.5"
    )


def test_documentation_range_counts_as_public() -> None:
    assert is_public_ip(ipaddress.ip_address("203.0.113.9"))
    assert not is_public_ip(ipaddress.ip_address("100.100.1.1"))


def test_IT_ADM_11_ipv6_same_64_shares_bucket() -> None:
    a = ipaddress.ip_address("2001:db8:1:2::1")
    b = ipaddress.ip_address("2001:db8:1:2:ffff::9")
    c = ipaddress.ip_address("2001:db8:1:3::1")
    assert client_net(a) == client_net(b) != client_net(c)
    k = b"k" * 32
    assert ip_hmac(k, a) == ip_hmac(k, b) != ip_hmac(k, c)
    assert ip_hmac(k, None) is None


def test_U_ADM_03_eta() -> None:
    assert eta_s(3, 240, 1) == 720
    assert eta_s(3, 240, 2) == 360


def test_U_ADM_04_ws_url() -> None:
    base = "https://sevi.tail1234.ts.net"
    sid = "abc"
    assert ws_url(base, "sevi.tail1234.ts.net", True, sid) == (
        "wss://sevi.tail1234.ts.net/web/stream?sid=abc"
    )
    assert ws_url(base, "a.trycloudflare.com", True, sid) == (
        "wss://a.trycloudflare.com/web/stream?sid=abc"
    )
    assert ws_url(base, "evil.example", True, sid).startswith("wss://sevi.tail1234.ts.net/")
    assert ws_url(base, "a.trycloudflare.com", False, sid).startswith("wss://sevi.tail1234")
    assert ws_url("http://127.0.0.1:8000", None, False, sid) == (
        "ws://127.0.0.1:8000/web/stream?sid=abc"
    )
