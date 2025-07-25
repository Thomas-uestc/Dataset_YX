# run_analysis.py

import argparse
import json
from pathlib import Path
import pandas as pd
import base64
import os
import re
import pyarrow.parquet as pq
import pyarrow as pa
import glob
from ._4_cot_reinstruction_generator import MultiModalAnalysisGenerator

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-generate CoT analysis and revised edit instructions from image records"
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="OpenAI API key"
    )
    parser.add_argument("--input-jsonal-dir", default='./output/_3_step_image', help="Path to difference JSONL file")
    parser.add_argument("--output-dir", default="./output/_4_cot_reinstruction", help="Output JSONL dir, which will include a batch of generated results")


    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI ChatCompletion model to use"
    )
    parser.add_argument("--input-parquet-dir", required=True, help="Input file containing a batch of parquet file, which include the source and target images")
    parser.add_argument("--source-column-name", default="src_img", help="Column name for source image bytes")
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

def main():
    args = parse_args()

    generator = MultiModalAnalysisGenerator(
        api_key=args.api_key,
        model=args.model
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
        print(f"Generating from original dataset_{parquet_path_number_str}.parquet for cot and reinstruction\n")
        print(f"The information of processing parquet is:\n")
        print(df.info(),"\n")

        # Read the input JSONL file into list of records
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
                # Grab the edit key 
                edit_text = (
                    rec["edit"]
                    if isinstance(rec["edit"], str)
                    else json.dumps(rec["edit"], ensure_ascii=False) # if the content of key "difference" is a dict
                )

                # Conver the step edited image saved in jsonal file from baed64 format into bytes format
                step_image_b64 = rec['step_edited']
                step_image =base64.b64decode(step_image_b64)

                # The input images are under bytes format
                cot_reediting_str = generator.generate(step_image, source_image, edit_text)
                try:
                    # Attempt to parse the difference string as JSON format
                    cot_reediting_str_clean = clean_json_block(cot_reediting_str)
                    cot_reediting = json.loads(cot_reediting_str_clean)
                except json.JSONDecodeError:
                    cot_reediting = cot_reediting_str
                # Add editing instructions to the record json and save as new output
                rec["CoT_Reedit"] = cot_reediting
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
