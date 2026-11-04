"""snap2read HTTPS 자체서명 인증서 생성기 — SAN에 현재 PC의 로컬 IP를 자동 포함.

실행:  python generate_cert.py
 (PC의 IP가 바뀌었거나 인증서가 만료됐을 때 다시 실행하면 됩니다)
"""
import datetime
import ipaddress
import re
import subprocess
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

BASE = Path(__file__).resolve().parent


def main():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.timezone.utc)

    # 현재 PC의 IPv4 수집 → SAN에 포함 (폰이 접속할 주소)
    out = subprocess.run(["ipconfig"], capture_output=True).stdout.decode("cp949", "replace")
    ips = re.findall(r"IPv4 주소[ .]*: ([\d.]+)", out) or \
          re.findall(r"IPv4 Address[ .]*: ([\d.]+)", out)
    ip_objs = [ipaddress.IPv4Address(ip) for ip in ips
               if not ip.startswith("169.254")]          # APIPA(연결 안 됨) 제외

    sans = [x509.DNSName("localhost"), x509.DNSName("snap2read.local")]
    sans += [x509.IPAddress(ip) for ip in ip_objs]

    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "snap2read")]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "snap2read")]))
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))

    (BASE / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (BASE / "key.pem").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    print("인증서 생성 완료 →", BASE / "cert.pem")
    print("SAN:", ["localhost", "snap2read.local"] + [str(ip) for ip in ip_objs])


if __name__ == "__main__":
    main()
