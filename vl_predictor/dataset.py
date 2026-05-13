import json
import re
from collections import defaultdict

import torch
from torch.utils.data import Dataset


def is_fake_name(name: str) -> bool:
    clean = re.sub(r"[\s\-_().,]", "", name.strip().lower())
    return clean in ["nn", "nnn", "nnnn", "n.n", "n.n.", "no name", "noname", "??", "???", "project", "nameless"]


def build_dataset(
    ascents_path: str = "data/ascents.jsonl",
    boulders_path: str = "data/boulders.jsonl",
    output_path: str = "data/dataset.pt",
    mode: str = "per_crag",
):
    boulders_by_id = {}
    with open(boulders_path) as f:
        for line in f:
            b = json.loads(line)
            if is_fake_name(b["boulder_name"]):
                continue
            boulders_by_id[b["boulder_id"]] = b

    climber_ids = set()
    boulder_ids_with_data = set()
    crag_slugs = set()
    sector_keys = set()
    positives = []
    climber_visited_boulders = defaultdict(set)
    climber_visited_crags = defaultdict(set)

    with open(ascents_path) as f:
        for line in f:
            a = json.loads(line)
            cid = a["climber_id"]
            if cid is None:
                continue
            ascent_type = a["ascent_type"]
            if ascent_type == "toprope":
                continue

            bid = a["boulder_id"]
            if bid not in boulders_by_id:
                continue

            boulder = boulders_by_id[bid]
            crag = boulder["crag_slug"]
            sector = boulder.get("sector_slug") or ""
            sector_key = (crag, sector)

            climber_ids.add(cid)
            boulder_ids_with_data.add(bid)
            crag_slugs.add(crag)
            sector_keys.add(sector_key)

            climber_visited_boulders[cid].add(bid)
            climber_visited_crags[cid].add(crag)

            label = {"go": 1, "send": 2, "flash": 3, "onsight": 3}[ascent_type]
            positives.append((cid, bid, label))

    climber_to_idx = {c: i for i, c in enumerate(sorted(climber_ids))}
    boulder_to_idx = {b: i for i, b in enumerate(sorted(boulder_ids_with_data))}
    crag_to_idx = {c: i for i, c in enumerate(sorted(crag_slugs))}

    boulder_to_crag = {
        bid: boulders_by_id[bid]["crag_slug"]
        for bid in boulder_ids_with_data
    }

    crag_to_boulders = defaultdict(list)
    for bid in boulder_ids_with_data:
        crag_to_boulders[boulder_to_crag[bid]].append(bid)

    n_climbers = len(climber_to_idx)
    n_boulders = len(boulder_to_idx)

    print(f"Climbers: {n_climbers}, Boulders: {n_boulders}, Crags: {len(crag_to_idx)}")
    print(f"Positive samples: {len(positives)}")
    label_counts = defaultdict(int)
    for _, _, lbl in positives:
        label_counts[lbl] += 1
    print(f"Label distribution: fail={label_counts[1]}, send={label_counts[2]}, flash={label_counts[3]}")

    # Build positive samples as tensors
    p_climber = torch.tensor([climber_to_idx[c] for c, _, _ in positives], dtype=torch.long)
    p_boulder = torch.tensor([boulder_to_idx[b] for _, b, _ in positives], dtype=torch.long)
    p_label = torch.tensor([l for _, _, l in positives], dtype=torch.long)

    # Build negative samples: all (climber, boulder) pairs not in positives
    n_climber_list = []
    n_boulder_list = []

    if mode == "per_crag":
        for cid in climber_ids:
            ci = climber_to_idx[cid]
            visited = climber_visited_boulders[cid]
            crags = climber_visited_crags[cid]

            candidate_boulders = set()
            for crag in crags:
                candidate_boulders.update(crag_to_boulders[crag])

            unvisited = list(candidate_boulders - visited)
            for bid in unvisited:
                n_climber_list.append(ci)
                n_boulder_list.append(boulder_to_idx[bid])
    else:
        all_boulder_idxs = list(range(n_boulders))
        for cid in climber_ids:
            ci = climber_to_idx[cid]
            visited = {boulder_to_idx[b] for b in climber_visited_boulders[cid]}

            for bi in all_boulder_idxs:
                if bi not in visited:
                    n_climber_list.append(ci)
                    n_boulder_list.append(bi)

    n_climber_t = torch.tensor(n_climber_list, dtype=torch.long)
    n_boulder_t = torch.tensor(n_boulder_list, dtype=torch.long)
    n_label = torch.zeros(len(n_climber_list), dtype=torch.long)

    print(f"Negative samples: {len(n_climber_list)}")

    sector_to_idx = {key: i for i, key in enumerate(sorted(sector_keys))}
    # Map boulder_idx → crag_idx / sector_idx for hierarchical priors.
    sorted_boulders = sorted(boulder_to_idx.items(), key=lambda kv: kv[1])
    boulder_crag_idx = torch.tensor(
        [crag_to_idx[boulder_to_crag[bid]] for bid, _ in sorted_boulders],
        dtype=torch.long,
    )
    boulder_sector_idx = torch.tensor(
        [
            sector_to_idx[(boulders_by_id[bid]["crag_slug"], boulders_by_id[bid].get("sector_slug") or "")]
            for bid, _ in sorted_boulders
        ],
        dtype=torch.long,
    )
    print(f"Sectors: {len(sector_to_idx)}")

    torch.save(
        {
            "p_climber": p_climber,
            "p_boulder": p_boulder,
            "p_label": p_label,
            "n_climber": n_climber_t,
            "n_boulder": n_boulder_t,
            "n_label": n_label,
            "climber_to_idx": climber_to_idx,
            "boulder_to_idx": boulder_to_idx,
            "crag_to_idx": crag_to_idx,
            "sector_to_idx": sector_to_idx,
            "boulder_crag_idx": boulder_crag_idx,
            "boulder_sector_idx": boulder_sector_idx,
            "n_climbers": n_climbers,
            "n_boulders": n_boulders,
            "n_crags": len(crag_to_idx),
            "n_sectors": len(sector_to_idx),
        },
        output_path,
    )
    print(f"Dataset saved to {output_path}")


class BoulderingDataset(Dataset):
    def __init__(self, dataset_path: str = "data/dataset.pt"):
        data = torch.load(dataset_path, weights_only=True)

        self.p_climber = data["p_climber"]
        self.p_boulder = data["p_boulder"]
        self.p_label = data["p_label"]
        self.n_climber = data["n_climber"]
        self.n_boulder = data["n_boulder"]

        self.n_pos = len(self.p_climber)
        self.n_neg = len(self.n_climber)
        self.total = self.n_pos + self.n_neg

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if idx < self.n_pos:
            return (
                self.p_climber[idx],
                self.p_boulder[idx],
                self.p_label[idx],
            )
        else:
            ni = idx - self.n_pos
            return (
                self.n_climber[ni],
                self.n_boulder[ni],
                torch.tensor(0, dtype=torch.long),
            )
