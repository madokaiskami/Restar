from huggingface_hub import create_repo, upload_folder

repo_id = "Aurelianous/restar_v1.0_model"  # Replace with your own repo
local_model_dir = r"outputs\restar_v1_0\model"

create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
upload_folder(
    folder_path=local_model_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="v1.0 epoch3 final",
)
print("pushed to:", repo_id)
