import argparse
import json
import base64
import pandas as pd
from pathlib import Path
from typing import Any
import os
import re
import pyarrow.parquet as pq
import pyarrow as pa
import glob
from ._1_difference_generator import DifferenceDescriptionGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate image difference descriptions using OpenAI API")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    #parser.add_argument("--source", nargs='+', required=True, help="List of source image paths")
    #parser.add_argument("--target", nargs='+', required=True, help="List of target image paths")
    parser.add_argument("--output-dir", default="./output/_1_difference", help="Output JSONL dir, which will include a batch of generated results")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")

    # for parquet file input
    parser.add_argument("--input-parquet-dir", required=True, help="Input file containing a batch of parquet file, which include the source and target images")
    parser.add_argument("--source-column-name", default="src_img", help="Column name for source image bytes")
    parser.add_argument("--target-column-name", default="edited_img", help="Column name for target image bytes")
    return parser.parse_args()

def clean_json_block(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json\n", "", 1).replace("\n", "", 1) if s.startswith("json") else s
    return s

def extract_index_number_int(path):
    # train-00012-xxx.parquet，extract int(12)
    m = re.search(r'train-(\d+)-', os.path.basename(path))
    return int(m.group(1)) if m else -1
def extract_index_number_str(path):
    # train-00012-xxx.parquet，extract str(00012)
    m = re.search(r'train-(\d+)-', os.path.basename(path))
    return m.group(1) if m else -1

def main() -> None:
    args = parse_args()
    # Initialize the DifferenceDescriptionGenerator
    generator = DifferenceDescriptionGenerator(args.api_key, model=args.model)

    parquet_dir = args.input_parquet_dir
    all_paths = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    valid_paths = [
        p for p in all_paths
        if not os.path.basename(p).startswith('._')
    ]
    
    # Sequence the path list with order of number: train-00000-xxx, train-00001-xxx, ...
    valid_paths.sort(key=extract_index_number_int)
    length = len(valid_paths)
    print(f"The number of parquet is {length} ")

    for parquet_path in valid_paths:
        
        # Extract the sequence string, for convenient
        parquet_path_number_str = extract_index_number_str(parquet_path)

        # Read a parquet file
        df = pd.read_parquet(parquet_path)
        print(f"Generating from original dataset_{parquet_path_number_str}.parquet for difference\n")
        print(f"The information of processing parquet is:\n")
        print(df.info(),"\n")
        # Convert the specified columns to lists
        source_images = [x['bytes'] for x in df[args.source_column_name]]
        target_images = [x['bytes'] for x in df[args.target_column_name]]
        source_images_name = [x['path'] for x in df[args.source_column_name]]
        target_images_name = [x['path'] for x in df[args.target_column_name]]

        # Call the generator 
        differences = generator.process_batch(source_images, target_images)
        # Write the differences into json
        output_path = f"{args.output_dir}/{parquet_path_number_str}.jsonal"
        output_path = Path(output_path)
        with output_path.open("w", encoding="utf-8") as f:
            for src_name, tgt_name, diff_str in zip(source_images_name, target_images_name, differences):
                try:
                    # Attempt to parse the difference string as JSON format
                    diff_str_clean = clean_json_block(diff_str)
                    diff = json.loads(diff_str_clean)
                except json.JSONDecodeError:
                    # Fallback to raw string if the API does not return valid JSON
                    diff = diff_str
                record = {
                    "source": src_name,
                    "target": tgt_name,
                    "difference": diff,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
