import asyncio
import os
import json
import unstructured_client
from unstructured_client.models import shared, errors

client = unstructured_client.UnstructuredClient(
    api_key_auth="aOFAWogpEK4rnFiV6jU58bTiFwHVe7"
)

async def partition_file_via_api(filename):
    req = {
        "partition_parameters": {
            "files": {
                "content": open(filename, "rb"),
                "file_name": os.path.basename(filename),
            },
            "strategy": shared.Strategy.AUTO,
            "vlm_model": "gpt-4o",
            "vlm_model_provider": "openai",
            "languages": ['eng'],
            "split_pdf_page": True, 
            "split_pdf_allow_failed": True,
            "split_pdf_concurrency_level": 15
        }
    }

    try:
        res = await client.general.partition_async(request=req)
        return res.elements
    except errors.UnstructuredClientError as e:
        print(f"Error partitioning {filename}: {e.message}")
        return []

async def process_file_and_save_result(input_filename, output_dir):
    elements = await partition_file_via_api(input_filename)

    if elements:
        results_name = f"{os.path.basename(input_filename)}.json"
        output_filename = os.path.join(output_dir, results_name)

        with open(output_filename, "w") as f:
            json.dump(elements, f)

def load_filenames_in_directory(input_dir):
    filenames = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith('.json'):
                filenames.append(os.path.join(root, file))

    return filenames

async def process_files():
    # Initialize with either a directory name, to process everything in the dir,
    # or a comma-separated list of filepaths.
    input_dir = "./FileFolder"   # "path/to/input/directory"
    input_files = None # "path/to/file,path/to/file,path/to/file"

    # Set to the directory for output json files. This dir 
    # will be created if needed.
    output_dir = "./output/"

    if input_dir:
        filenames = load_filenames_in_directory(input_dir)
    else:
        filenames = input_files.split(",")

    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for filename in filenames:
        tasks.append(
            process_file_and_save_result(filename, output_dir)
        )

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(process_files())