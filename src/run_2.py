import argparse
import json
from pathlib import Path
import pandas as pd
from ._2_instruction_generator import EditInstructionGenerator
import os
import re
import pyarrow.parquet as pq
import pyarrow as pa
import glob

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate editing instructions from difference descriptions"
    )
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--input-jsonal-dir", default='./output/_1_difference', help="Path to difference JSONL file")
    parser.add_argument("--input-parquet-dir", required=True, help="Input file containing a batch of parquet file, which include the source and target images")
    parser.add_argument("--source-column-name", default="src_img", help="Column name for source image bytes")
    parser.add_argument("--output-dir", default="./output/_2_instruction", help="Output JSONL dir, which will include a batch of generated results")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
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
    generator = EditInstructionGenerator(args.api_key, model=args.model)

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
        print(f"Generating from original dataset_{parquet_path_number_str}.parquet for instruction\n")
        print(f"The information of processing parquet is:\n")
        print(df.info(),"\n")
        # Convert the specified columns to lists
        source_images = [x['bytes'] for x in df[args.source_column_name]]

        
        # Read the input JSONL file into list of records
        input_path = f"{args.input_jsonal_dir}/{parquet_path_number_str}.jsonal"
        input_path = Path(input_path)
        records = []
        with input_path.open("r", encoding="utf-8") as f:
            # Read each line from json file
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        output_path = f"{args.output_dir}/{parquet_path_number_str}.jsonal"
        output_path = Path(output_path)
        with output_path.open("w", encoding="utf-8") as f:
            for rec, source_image in zip(records, source_images ): # source_image is corresponding image bytes
                # Grab the difference key 
                diff_text = (
                    rec["difference"]
                    if isinstance(rec["difference"], str)
                    else json.dumps(rec["difference"], ensure_ascii=False) # if the content of key "difference" is a dict
                )

                instr_str = generator.generate_instructions(source_image, diff_text)
                try:
                    # Attempt to parse the difference string as JSON format
                    instr_str_clean = clean_json_block(instr_str)
                    instr = json.loads(instr_str_clean)
                except json.JSONDecodeError:
                    instr = instr_str
                # Add editing instructions to the record json and save as new output
                rec["edit"] = instr
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
