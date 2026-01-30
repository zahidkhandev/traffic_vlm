import os

from huggingface_hub import snapshot_download

# 1. Force standard HTTP (more stable through proxies than the "fast" rust downloader)
os.environ["HF_HUB_DISABLE_HF_TRANSFER"] = "1"

# 2. Increase timeout significantly (default is often too short for corp proxies)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

print("--- STARTING ROBUST DOWNLOAD ---")
print("Model: llava-hf/llava-v1.6-mistral-7b-hf")
print("This may take time. If it fails, run it again - it WILL resume.")

try:
    path = snapshot_download(
        repo_id="llava-hf/llava-v1.6-mistral-7b-hf",
        resume_download=True,
        local_files_only=False,
        # If your CNTLM is confirmed at 127.0.0.1:8080 (from your logs), set it here explicitly
        proxies={"http": "http://127.0.0.1:8080", "https": "http://127.0.0.1:8080"},
    )
    print(f"\nSUCCESS! Model downloaded to: {path}")
except Exception as e:
    print(f"\nDOWNLOAD FAILED: {e}")
    print("Run this script again to resume.")
