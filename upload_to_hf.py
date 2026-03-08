#!/usr/bin/env python3
"""Upload Melee RL checkpoints and key files to Hugging Face."""

from huggingface_hub import HfApi, login

REPO_ID = "melee-rl-fox-mango"

FILES_TO_UPLOAD = [
    ("checkpoints/mango_final.pt", "mango_final.pt"),
    ("checkpoints/puff_final.pt", "puff_final.pt"),
    ("checkpoints/dolphin_fox_final_501760.pt", "dolphin_fox_final_501760.pt"),
    ("rewards/competitive.py", "rewards/competitive.py"),
    ("rewards/puff.py", "rewards/puff.py"),
    ("emulator_env/policy_runner.py", "emulator_env/policy_runner.py"),
    ("emulator_env/models.py", "emulator_env/models.py"),
    ("README.md", "README.md"),
    ("emulator_env/README.md", "emulator_env/README.md"),
]

def main():
    print("Step 1: Logging in to Hugging Face...")
    print("Get your token from: https://huggingface.co/settings/tokens")
    login()

    api = HfApi()
    username = api.whoami()["name"]
    full_repo_id = f"{username}/{REPO_ID}"

    print(f"\nStep 2: Creating repo '{full_repo_id}'...")
    try:
        api.create_repo(full_repo_id, repo_type="model", exist_ok=True)
        print(f"Repo ready: https://huggingface.co/{full_repo_id}")
    except Exception as e:
        print(f"Repo creation: {e}")

    print(f"\nStep 3: Uploading files...")
    for local_path, repo_path in FILES_TO_UPLOAD:
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=full_repo_id,
            )
            print(f"  Uploaded: {local_path} -> {repo_path}")
        except FileNotFoundError:
            print(f"  SKIPPED (not found): {local_path}")
        except Exception as e:
            print(f"  ERROR uploading {local_path}: {e}")

    print(f"\nDone! View at: https://huggingface.co/{full_repo_id}")


if __name__ == "__main__":
    main()
