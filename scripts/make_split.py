from collections import Counter

from neural_denoise.cache import build_crop_cache
from neural_denoise.config import (
    BLOCK_SIZE,
    PATCH_SIZE,
    PATCHES_PER_IMAGE,
    SPLIT_PATH,
    SPLIT_SEED,
    VALIDATION_CROPS_PER_IMAGE,
)
from neural_denoise.datasets import find_pairs
from neural_denoise.splits import make_split, save_split


def main() -> None:
    pairs = find_pairs()
    split = make_split([pair.instance.name for pair in pairs])
    save_split(split, SPLIT_PATH)
    by_name = {pair.instance.name: pair for pair in pairs}
    for name, names in split.items():
        phones = Counter(by_name[item].instance.phone for item in names)
        print(f"{name}: {len(names)} instances, phones {dict(sorted(phones.items()))}")
    train = [by_name[name] for name in split["train"]]
    validation = [by_name[name] for name in split["validation"]]
    print(f"Caching {len(train) * PATCHES_PER_IMAGE} training patches of {PATCH_SIZE} px")
    build_crop_cache("train", train, PATCH_SIZE, PATCHES_PER_IMAGE, SPLIT_SEED)
    print(f"Caching {len(validation) * VALIDATION_CROPS_PER_IMAGE} validation crops")
    build_crop_cache(
        "validation", validation, BLOCK_SIZE, VALIDATION_CROPS_PER_IMAGE, SPLIT_SEED + 1
    )
    build_crop_cache("calibration", train, BLOCK_SIZE, 1, SPLIT_SEED + 2)


if __name__ == "__main__":
    main()
