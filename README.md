# Dataset_YX

This repository provides a small pipeline for generating training data using the OpenAI API.



## Installation

```bash
git clone https://github.com/Thomas-uestc/Dataset_YX.git
cd Dataset_YX
conda create -n GD python
conda activate GD
pip install -r requirements.txt
```

## Downloading Original Dataset
Do not need to change following scripts.
```bash
# The size of each file ≈ 450M
wget -O ./dataset/train-00000-of-00105.parquet "https://gla-my.sharepoint.com/personal/2840046d_student_gla_ac_uk/_layouts/52/download.aspx?share=EflfoQ9-WUVCr3lmUzR2BEEBMh3Kn6xAThtX5sJ5EFudhw"
wget -O ./dataset/train-00001-of-00105.parquet "https://gla-my.sharepoint.com/personal/2840046d_student_gla_ac_uk/_layouts/52/download.aspx?share=ESOHtTTpUvZGn9m6Tt_BII4BzvRuIYH1haIeBWNMuc_90Q"
```

## Usage

### Difference Generator
Take source_img and target_img into the `gpt-4o` and generated corresponding difference descriptions.

```bash
# Input file containing a batch of parquet file, which include the source and target images bytes.
python -m src.run_1 \
--input-parquet-dir ./dataset \
--api-key YOUR_API_KEY 
```

The output saved in `./output/_1_difference/xxxxx.jsonl` 's construction:
```bash
{
  "source": "the name of source img",
  "target": "the name of corresponding target img",
  "difference": {
                  "1": "Difference Description",
                  "2": "Difference Description",
                  ...
                  }
}
```

### Editing Instruction Generator

After generating differences, run the second script to utilize `gpt-4o` to generate editing instruction.

```bash
# Input file containing a batch of parquet file, which include the source and target images bytes.
python -m src.run_2 \
--input-parquet-dir ./dataset \
--api-key YOUR_API_KEY 
```
The output saved in `./output/_2_instruction/xxxxx.jsonl` will be:
```bash
{
  "source": "the name of source img",
  "target": "the name of corresponding target img",
  "difference": {
                  "1": "Difference Description",
                  "2": "Difference Description",
                  ...
                  },
  "edit": {
            "1": "Editing Action",
            "2": "Editing Action",
            ...
          }
}
```

### Step Image Generator

After generating differences and editing instructions, run the third script to utilize `gpt-image-1` to generate step editing image.

```bash
# Input file containing a batch of parquet file, which include the source and target images bytes.
python -m src.run_3 \
--input-parquet-dir ./dataset \
--api-key YOUR_API_KEY 
```

The output saved in `./output/_3_step_image/xxxxx.jsonl` will be:
```bash
{
  "source": "the name of source img",
  "target": "the name of corresponding target img",
  "difference": {
                  "1": "Difference Description",
                  "2": "Difference Description",
                  ...
                  },
  "edit": {
            "1": "Editing Action",
            "2": "Editing Action",
            ...
          },
  "step_edited": "base64string of step edited img",
}
```

### CoT and Re_Instruction Generator

After generating differences, editing instrctions, and step edited image, run the fourth script to utilize `gpt-4o` to generate CoT and Re_Instruction.

```bash
# Input file containing a batch of parquet file, which include the source and target images bytes.
python -m src.run_4 \
--input-parquet-dir ./dataset \
--api-key YOUR_API_KEY 
```

The output saved in `./output/_4_cot_reinstruction/xxxxx.jsonl` will be:
```bash
{
  "source": "the name of source img",
  "target": "the name of corresponding target img",
  "difference": {
                  "1": "Difference Description",
                  "2": "Difference Description",
                  ...
                  },
  "edit": {
            "1": "Editing Action",
            "2": "Editing Action",
            ...
          },
  "step_edited": "base64string of step edited img",
  "CoT_Reedit": {
                  "CoT_1": "Chain of Thought",
                  "CoT_2": "Chain of Thought",
                  ...
                  "Re_Edit_1: "editing instuction",
                  "Re_Edit_2: "editing instuction",
                  ...
                },
}
```

## Result_Example

You can check `./output/example` to observe the desired four output jsonal of four generator corresponding.
