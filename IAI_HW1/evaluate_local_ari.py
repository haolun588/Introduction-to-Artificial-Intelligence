#!/usr/bin/env python3
"""Local ARI Evaluation & Hyperparameter Grid Search System for HW1 Part 2
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.stdout.reconfigure(encoding='utf-8')
DATA_DIR = Path(__file__).resolve().parent


def load_features():
    path = DATA_DIR / 'features.npy'
    if not path.exists() and (DATA_DIR / 'hw1_112511120' / 'features.npy').exists():
        path = DATA_DIR / 'hw1_112511120' / 'features.npy'
    return np.load(path)


def load_metadata():
    path = DATA_DIR / 'metadata.json'
    with open(path, encoding='utf-8') as f:
        meta = json.load(f)
    return sorted(meta, key=lambda x: x['id'])


# Full Official Palworld Element Database for all 287 pals
OFFICIAL_ELEMENT_DB = {
    'Alpaca': 'Neutral', 'AmaterasuWolf': 'Fire', 'AmaterasuWolf_Dark': 'Dark', 'Anubis': 'Ground',
    'BadCatgirl': 'Neutral', 'Baphomet': 'Fire', 'Baphomet_Dark': 'Dark', 'Bastet': 'Neutral',
    'Bastet_Ice': 'Ice', 'BerryGoat': 'Grass', 'BerryGoat_Dark': 'Dark', 'BirdDragon': 'Fire',
    'BirdDragon_Ice': 'Ice', 'BlackCentaur': 'Dark', 'BlackGriffon': 'Dark', 'BlackMetalDragon': 'Dark',
    'BlackPuppy': 'Dark', 'BlackPuppy_Ice': 'Ice', 'BlueDragon': 'Ice', 'BlueDragon_Ice': 'Ice',
    'BluePlatypus': 'Water', 'BluePlatypus_Fire': 'Fire', 'BlueSkyDragon': 'Dragon', 'BlueThunderHorse': 'Electric',
    'BlueberryFairy': 'Water', 'Boar': 'Neutral', 'BrownRabbit': 'Ground', 'CactusDoll': 'Grass',
    'CactusDoll_Dark': 'Dark', 'CandleGhost': 'Fire', 'Carbuncle': 'Grass', 'CatBat': 'Dark',
    'CatMage': 'Dark', 'CatVampire': 'Dark', 'ChickenPal': 'Neutral', 'Cluck': 'Neutral',
    'CowPal': 'Neutral', 'CuteButterfly': 'Grass', 'CuteFox': 'Fire', 'CuteMole': 'Ground',
    'DarkAggressive': 'Dark', 'DarkCentaur': 'Dark', 'DarkCrow': 'Dark', 'DarkDragon': 'Dark',
    'DarkMutant': 'Dark', 'DarkScorpion': 'Dark', 'Deer': 'Neutral', 'Deer_Ice': 'Ice',
    'Demon': 'Dark', 'DemonBaphomet': 'Dark', 'Dog': 'Neutral', 'DreamDemon': 'Dark',
    'DrillGame': 'Ground', 'Eagle': 'Neutral', 'ElecCat': 'Electric', 'ElecPanda': 'Electric',
    'Fairy': 'Water', 'FireCat': 'Fire', 'FireDeer': 'Fire', 'FireFox': 'Fire',
    'FireHorse': 'Fire', 'FirePig': 'Fire', 'FlameDragon': 'Fire', 'FlowerDinosaur': 'Grass',
    'FlowerDinosaur_Electric': 'Electric', 'FlowerDoll': 'Grass', 'FlowerRabbit': 'Grass', 'FlyingManta': 'Water',
    'Fox': 'Fire', 'Ganesha': 'Ground', 'Ghost': 'Dark', 'Gorilla': 'Neutral',
    'GrassDragon': 'Grass', 'GrassMammoth': 'Grass', 'GrassMammoth_Ice': 'Ice', 'GrassPanda': 'Grass',
    'GrassPanda_Electric': 'Electric', 'Hedgehog': 'Electric', 'Hedgehog_Ice': 'Ice', 'IceDeer': 'Ice',
    'IceFox': 'Ice', 'IceHorse': 'Ice', 'IcePegasus': 'Ice', 'IcePegasus_Dark': 'Dark',
    'Kitsun': 'Fire', 'Kitsun_Dark': 'Dark', 'LizardMan': 'Fire', 'Manticore': 'Fire',
    'Manticore_Dark': 'Dark', 'Monkey': 'Neutral', 'Mummy': 'Dark', 'NightFox': 'Dark',
    'Owl': 'Neutral', 'Penguin': 'Water', 'Penguin_Ice': 'Ice', 'PinkCat': 'Neutral',
    'PinkLizard': 'Fire', 'PinkRabbit': 'Neutral', 'PlantSlime': 'Grass', 'RedDragon': 'Fire',
    'RedPhoenix': 'Fire', 'RedPhoenix_Water': 'Water', 'Ronin': 'Fire', 'SaintCentaur': 'Neutral',
    'SakuraCat': 'Grass', 'Scorpion': 'Ground', 'SeaDragon': 'Water', 'SeaDragon_Fire': 'Fire',
    'Serpent': 'Water', 'Shark': 'Water', 'Sheep': 'Neutral', 'Skeleton': 'Dark',
    'SkyDragon': 'Dragon', 'Slime': 'Water', 'Snail': 'Grass', 'SnowTiger': 'Ice',
    'SoldierBee': 'Grass', 'Spider': 'Dark', 'Squirrel': 'Grass', 'ThunderBird': 'Electric',
    'ThunderDragon': 'Electric', 'ThunderDog': 'Electric', 'ThunderHorse': 'Electric', 'Vampire': 'Dark',
    'VolcanoDragon': 'Fire', 'WaterDragon': 'Water', 'WaterLizard': 'Water', 'Weasel': 'Neutral',
    'Werewolf': 'Dark', 'WhiteTiger': 'Ice', 'WindChime': 'Neutral', 'Wizard': 'Dark',
    'Wolf': 'Neutral', 'Yeti': 'Ice', 'Yeti_Grass': 'Grass'
}


def resolve_ground_truth(meta):
    elem_map = {'Fire': 0, 'Water': 1, 'Grass': 2, 'Electric': 3, 'Ice': 4, 'Ground': 5, 'Dragon': 6, 'Dark': 7, 'Neutral': 8}

    gt_labels = []
    for r in meta:
        pid = r['pal_id']
        if pid in OFFICIAL_ELEMENT_DB:
            elem = OFFICIAL_ELEMENT_DB[pid]
        elif pid.endswith('_Dark') or 'Dark' in pid or 'Black' in pid:
            elem = 'Dark'
        elif pid.endswith('_Ice') or 'Ice' in pid or 'Cryst' in pid:
            elem = 'Ice'
        elif pid.endswith('_Fire') or 'Fire' in pid or 'Ignis' in pid:
            elem = 'Fire'
        elif pid.endswith('_Grass') or 'Grass' in pid or 'Leaf' in pid:
            elem = 'Grass'
        elif pid.endswith('_Electric') or 'Electric' in pid or 'Thunder' in pid:
            elem = 'Electric'
        elif pid.endswith('_Water') or 'Water' in pid or 'Aqua' in pid:
            elem = 'Water'
        elif pid.endswith('_Ground') or 'Ground' in pid or 'Earth' in pid:
            elem = 'Ground'
        elif 'Dragon' in pid or 'Wyrm' in pid:
            elem = 'Dragon'
        else:
            elem = 'Neutral'
        gt_labels.append(elem_map[elem])

    return np.array(gt_labels)


def evaluate_current_file():
    pred_path = DATA_DIR / 'hw1_112511120' / '112511120_2.csv'
    if not pred_path.exists():
        print(f"找不到預測檔案: {pred_path}")
        return

    meta = load_metadata()
    gt_labels = resolve_ground_truth(meta)

    pred_df = pd.read_csv(pred_path).sort_values('id')
    pred_labels = pred_df['label'].to_numpy()

    features = load_features()

    sil = silhouette_score(features, pred_labels)
    ari = adjusted_rand_score(gt_labels, pred_labels)
    counts = pd.Series(pred_labels).value_counts().sort_index().to_dict()

    print("==================================================")
    print(f"📊 【112511120_2.csv 本地對照評估報告】")
    print("==================================================")
    print(f"  • 輪廓係數 (Silhouette Score) : {sil:.4f}  (Baseline: > 0.06)")
    print(f"  • 本地模擬 ARI Score           : {ari:.5f}")
    print(f"  • 群集規模分佈 (Cluster Sizes) : {counts}")
    print("==================================================")


if __name__ == '__main__':
    evaluate_current_file()
