import os

import google.cloud.storage as storage

def load_from_bucket(bucket_name: str, file_name: str, local_path: str) -> None:
    """
    Load the file from the bucket to the local path

    Parameters:
    -----------
    `bucket_name`: `str`
        name of the bucket
    `file_name`: `str`
        name of the file to be downloaded
    `local_path`: `str`
        path to the local file
    """
    # If client credentials are local, load from file
    if os.path.exists("gcp_keys.json"):
        client = storage.Client.from_service_account_json("gcp_keys.json")
    # Else, assume running in GCP and load from environment
    else:
        client = storage.Client()

    # Get bucket
    bucket = client.get_bucket(bucket_name)

    # List all blobs in the specified directory and its subdirectories
    blobs = bucket.list_blobs(prefix=file_name)

    for blob in blobs:
        # Create local directory structure
        local_file_path = local_path + blob.name[len(file_name):]
        local_directory = local_file_path.rsplit('/', 1)[0]

        # Ensure local directory exists
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)

        # Download the blob to the local path
        blob.download_to_filename(local_file_path)
