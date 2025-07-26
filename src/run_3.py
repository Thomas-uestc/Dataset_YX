# run_step_edit.py

import argparse
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import io
import base64
import pyarrow.parquet as pq
import pyarrow as pa
import glob
import re
import os
from ._3_step_image_generator import StepImageEditor

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-apply 'step' edits to images and update JSONL records"
    )
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--input-jsonal-dir", default='./output/_2_instruction', help="Path to instruction JSONL file")
    parser.add_argument("--input-parquet-dir", required=True, help="Input file containing a batch of parquet file, which include the source and target images")
    parser.add_argument("--source-column-name", default="src_img", help="Column name for source image bytes")
    parser.add_argument("--output-dir", default="./output/_3_step_image", help="Output JSONL dir, which will include a batch of generated results")
    parser.add_argument("--n",    type=int, default=1, help="Images per edit")
    return parser.parse_args()

def extract_index_number_int(path):
    # train-00012-xxx.parquet，extract int(12)
    m = re.search(r'train-(\d+)-', os.path.basename(path))
    return int(m.group(1)) if m else -1
def extract_index_number_str(path):
    # train-00012-xxx.parquet，extract str(00012)
    m = re.search(r'train-(\d+)-', os.path.basename(path))
    return m.group(1) if m else -1

def main():
    args = parse_args()

    editor = StepImageEditor(
        api_key=args.api_key,
        n=args.n,
    )

    parquet_dir = args.input_parquet_dir
    all_paths = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    valid_paths = [
        p for p in all_paths
        if not os.path.basename(p).startswith('._')
    ]
    
    # Sequence the path list with order of number: train-00000-xxx, train-00001-xxx, ...
    valid_paths.sort(key=extract_index_number_int)

    for parquet_path in valid_paths:

        # Extract the sequence string, for convenient
        parquet_path_number_str = extract_index_number_str(parquet_path)

        # Read the parquet file
        df = pd.read_parquet(parquet_path)
        # Convert the specified columns to lists
        source_images = [x['bytes'] for x in df[args.source_column_name]]

        print(f"Generating from original dataset_{parquet_path_number_str}.parquet for step images\n")
        print(f"The information of processing parquet is:\n")
        print(df.info(),"\n")

        # Read the JSONL file into list of records
        input_path = f"{args.input_jsonal_dir}/{parquet_path_number_str}.jsonal"
        input_path = Path(input_path)
        records = []
        with input_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        output_path = f"{args.output_dir}/{parquet_path_number_str}.jsonal"
        output_path = Path(output_path)
        with output_path.open("w", encoding="utf-8") as f:
            for rec, source_image in zip(records, source_images ): # source_image is corresponding image bytes
                # Grab the difference key 
                edit_text = (
                    rec["edit"]
                    if isinstance(rec["edit"], str)
                    else rec["edit"].get("1", "")  # Only take action 1 to execute
                )
                print("The specific action of this step edited image is: \n")
                print(edit_text)
                # Get the width and height of the source image
                img = Image.open(io.BytesIO(source_image))
                img_format = img.format
                width, height = img.size

                step_edited_img_bytes = editor.apply_step(source_image, edit_text, width, height, img_format)

                # Add step edited image base64 string to the record jsonal and save as new output
                # Attention: jsonal cannot accept bytes, so we need to encode the bytes to base64 string
                rec["step_edited"] = base64.b64encode(step_edited_img_bytes).decode("utf-8")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
