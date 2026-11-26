'''
Downloads the trained weights used by the demo, so you can run it without training
Run: python -m scripts.download_model
'''
import hashlib
import urllib.request
from pathlib import Path

import yaml

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    cfg = yaml.safe_load(open("config/config.yaml"))["model"]
    dest = Path(cfg["path"])

    if dest.exists() and sha256(dest) == cfg["sha256"]:
        print(f"{dest} already present and verified")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading {cfg['url']} ...")
    urllib.request.urlretrieve(cfg["url"], tmp)

    if sha256(tmp) != cfg["sha256"]:  # don't load a corrupted or tampered checkpoint
        tmp.unlink()
        raise SystemExit("Checksum mismatch: download is corrupted or the release changed")

    tmp.rename(dest)
    print(f"Saved to {dest}")

if __name__ == "__main__":
    main()
