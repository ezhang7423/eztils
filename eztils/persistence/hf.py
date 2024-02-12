"""
Requires setting HF_TOKEN env variable
"""

import os
from pathlib import Path

from huggingface_hub import create_repo, login
from huggingface_hub import upload_folder as hf_upload_folder


def upload_folder(local_folder: str, remote: str, username="ezipe"):
    """Upload a folder to a huggingface repo

    :param local_folder: This should be the data root. The folder will be uploaded to the repo root.
    Files already existing in the remote will overwritten.
    iles existing in the remote but not in the local will NOT be deleted.
    Files existing in the local but not in the remote will be uploaded.
    """
    login(token=os.environ["HF_TOKEN"])
    lf = Path(local_folder).resolve()
    try:
        create_repo(repo_id=remote)
    except:
        pass

    print(f"Uploading folder {lf} to huggingface {remote}...")
    hf_upload_folder(folder_path=lf, repo_id=f"{username}/{remote}")
