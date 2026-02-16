from huggingface_hub import upload_folder, create_repo, HfApi

repo_id = "harshachinthala/Iris_prediction_app"

print(f"Creating/Checking repo: {repo_id}...")
try:
    # First try with normal streamlit SDK
    create_repo(repo_id=repo_id, repo_type="space", space_sdk="streamlit", exist_ok=True)
    print("Repo created/verified with 'streamlit' SDK.")
except Exception as e:
    print(f"Failed to create with 'streamlit' SDK: {e}")
    try:
        # Fallback: try creating as static, then upload will overwrite config
        print("Fallback: Creating as 'static' space...")
        create_repo(repo_id=repo_id, repo_type="space", space_sdk="static", exist_ok=True)
        print("Repo created as static space.")
    except Exception as e2:
        print(f"Failed fallback creation: {e2}")

print("Uploading folder...")
try:
    upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space"
    )
    print("Upload successful!")
except Exception as e:
    print(f"Upload failed: {e}")
