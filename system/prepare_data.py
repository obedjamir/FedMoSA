"""Build every client's slices once (run from system/).

    python prepare_data.py -data liver_seg -nc 3

Writes ../dataset/<data>/slices_<client>_train_<id>_volumes.pkl (all slices of every training volume; each epoch
draws a fresh 25% of every volume) and slices_<client>_<split>_<id>.pkl for val / test (lists of (image uint8 HxW,
mask uint8 HxW, modality)). Training (main.py and the NVFlare example) loads these directly; main.py also builds a
missing file on first use.
"""
import argparse
import os

from utils.data_utils import read_client_data, subset_counts

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-data", "--dataset", required=True)
    ap.add_argument("-nc", "--num_clients", type=int, required=True)
    ap.add_argument("--dataset_id", default="0", help="the run index main.py uses (times), 0 for the first run")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    for idx in range(args.num_clients):
        for split in args.splits:
            if not os.path.exists(os.path.join("../dataset", args.dataset, f"{split}{args.dataset_id}", f"{idx}.npz")):
                continue
            data = read_client_data(args.dataset, 0, idx, args.dataset_id, split)
            if split == "train":
                total = {}
                for m, imgs, _ in data:
                    total[m] = total.get(m, 0) + len(imgs)
                print(f"client {idx} train: {len(data)} volumes, {sum(total.values())} slices {total}; "
                      f"per epoch {subset_counts(data)}")
                continue
            counts = {}
            for _, _, m in data:
                counts[m] = counts.get(m, 0) + 1
            print(f"client {idx} {split}: {len(data)} slices {counts}")
