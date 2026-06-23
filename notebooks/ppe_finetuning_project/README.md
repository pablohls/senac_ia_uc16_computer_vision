# SH17 YOLO Finetuning

## 1) Treinar (baseline completo)

Use o Python do ambiente do projeto (Poetry venv):

```bash
/home/senacgoon.local/202473567/.cache/pypoetry/virtualenvs/senac-ia-uc-16-computer-vision-lhRmZcUU-py3.12/bin/python notebooks/ppe_finetuning_project/finetune_sh17.py --epochs 100 --batch 16 --device 0
```

## 2) Validar checkpoint

```bash
/home/senacgoon.local/202473567/.cache/pypoetry/virtualenvs/senac-ia-uc-16-computer-vision-lhRmZcUU-py3.12/bin/python notebooks/ppe_finetuning_project/finetune_sh17.py --validate-only --model runs/detect/sh17_yolov8n_finetune/weights/best.pt --device 0
```

## 3) Classes SH17 (17)

0 person
1 ear
2 ear-mufs
3 face
4 face-guard
5 face-mask-medical
6 foot
7 tools
8 glasses
9 gloves
10 helmet
11 hands
12 head
13 medical-suit
14 shoes
15 safety-suit
16 safety-vest
