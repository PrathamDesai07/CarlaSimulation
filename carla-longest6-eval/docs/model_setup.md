# Model Setup Guide

## Download

Google Drive: https://drive.google.com/drive/folders/1yG9LbVtSLaneHKlB5GL5Vhzz5Miue5Me?usp=sharing

## Placement

```
models/
├── transfuser_official/
│   └── model.pth          # Official TransFuser checkpoint
└── crossvit_50/
    └── model50.pth         # CrossViT-Fusion checkpoint

transfuser/
├── transfuser-2022/         # Extract transfuser-2022.zip here
└── bi-attenfusion/          # CrossViT-Fusion code for model50.pth
```

## Switching Models

```bash
# Via CLI
python3 scripts/run_longest6_eval.py --model-path /app/models/transfuser_official/model.pth
python3 scripts/run_longest6_eval.py --model-path /app/models/crossvit_50/model50.pth

# Via env var
MODEL_PATH=/app/models/crossvit_50/model50.pth docker-compose run --rm client
```

## Important

- `model50.pth` uses `backbone: crossvit_fusion`
- Official TransFuser uses `backbone: transfuser`
- The evaluation script must load the correct architecture per model
