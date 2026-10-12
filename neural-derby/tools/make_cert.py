"""Make a self-signed certificate so the TEAM phone can use live video (browsers need HTTPS for the camera).

Usage: python tools/make_cert.py            (writes data/cert.pem and data/key.pem)
On the team phone, open https://<laptop ip>:8443/phone/?team=1 once and accept the warning.
"""
import datetime
import ipaddress
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from derby import config  # noqa: E402
from derby.server.app import lan_ip  # noqa: E402


def main():
    ip = lan_ip()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Neural Derby")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address(ip))]), critical=False)
            .sign(key, hashes.SHA256()))
    (config.DATA_DIR / "key.pem").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    (config.DATA_DIR / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print(f"wrote data/cert.pem and data/key.pem for {ip} and localhost")


if __name__ == "__main__":
    main()
