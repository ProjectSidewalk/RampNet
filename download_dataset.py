import json
from pathlib import Path
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REPO_ID = "projectsidewalk/rampnet-dataset"
OUTPUT_DIR = "./dataset"
NUM_PROC = 8
HF_TOKEN = None 

def save_example(example, output_dir: Path):
    """
    Processes a single example (row) from the Hugging Face dataset and saves it
    as a .jpg and .json file in the specified output directory.

    This function is designed to be used with the `dataset.map()` method.
    
    Args:
        example (dict): A dictionary representing one row of the dataset.
        output_dir (Path): The directory to save the files in (e.g., 'output/train').
    """
    try:
        pano_id = example.get('pano_id')
        if not pano_id:
            logging.warning("Skipping example because it has no 'pano_id'.")
            return

        metadata = {key: value for key, value in example.items() if key != 'image'}
        
        json_path = output_dir / f"{pano_id}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
            
        image = example['image']
        if image:
            image_path = output_dir / f"{pano_id}.jpg"
            image.save(image_path, format="JPEG", quality=95)
            
    except Exception as e:
        pano_id_str = example.get('pano_id', 'UNKNOWN_ID')
        logging.error(f"Failed to process pano_id '{pano_id_str}': {e}")


def main():
    """Main function to run the dataset recreation process."""
    output_path = Path(OUTPUT_DIR)
    
    logging.info(f"Loading dataset '{REPO_ID}' from the Hub...")
    ds_dict = load_dataset(REPO_ID, token=HF_TOKEN, num_proc=NUM_PROC)
    logging.info(f"Dataset loaded. Found splits: {list(ds_dict.keys())}")

    split_mapping = {
        "train": "train",
        "validation": "val",
        "test": "test"
    }

    for hf_split_name, local_folder_name in split_mapping.items():
        if hf_split_name not in ds_dict:
            logging.warning(f"Split '{hf_split_name}' not found in the dataset. Skipping.")
            continue
            
        logging.info(f"--- Processing split: {hf_split_name} ---")
        
        ds_split = ds_dict[hf_split_name]
        
        split_output_path = output_path / local_folder_name
        split_output_path.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Recreating files in: {split_output_path}")
        
        ds_split.map(
            save_example,
            fn_kwargs={'output_dir': split_output_path},
            num_proc=NUM_PROC,
            desc=f"Recreating {hf_split_name} split"
        )
        
        logging.info(f"--- Finished processing split: {hf_split_name} ---")
        
    logging.info(f"Dataset recreation complete! Files are saved in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()