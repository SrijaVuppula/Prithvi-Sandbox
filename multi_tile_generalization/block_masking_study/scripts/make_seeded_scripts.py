from pathlib import Path

def patch(src, dst, edits):
    s = Path(src).read_text()
    for old, new in edits:
        n = s.count(old)
        assert n == 1, f"{src}: expected exactly 1 match, found {n} for:\n{old}"
        s = s.replace(old, new)
    Path(dst).write_text(s)
    print(f"wrote {dst}  ({len(edits)} edits)")

patch("scripts/train_block_finetune.py", "scripts/train_block_finetune_seeded.py", [
 ('    args = ap.parse_args()\n',
  '    ap.add_argument("--seed", type=int, required=True,\n'
  '                    help="training seed: data order + mask/ratio stream")\n'
  '    args = ap.parse_args()\n'
  '    torch.manual_seed(args.seed)\n'),
 ('    ckpt_dir = CKPT_DIR / args.backbone\n',
  '    ckpt_dir = CKPT_DIR / "seeds" / f"seed{args.seed}" / args.backbone\n'),
 ('    LOG_DIR.mkdir(parents=True, exist_ok=True)\n',
  '    log_dir = LOG_DIR / "seeds" / f"seed{args.seed}"\n'
  '    log_dir.mkdir(parents=True, exist_ok=True)\n'),
 ('open(LOG_DIR / f"{args.backbone}_train_log.csv"',
  'open(log_dir / f"{args.backbone}_train_log.csv"'),
 ('open(LOG_DIR / f"{args.backbone}_finetune_cost.csv"',
  'open(log_dir / f"{args.backbone}_finetune_cost.csv"'),
 ('    rng = np.random.default_rng(2026)\n',
  '    rng = np.random.default_rng(args.seed)\n'),
])

patch("scripts/run_paired_block_random_finetuned.py", "scripts/run_paired_block_random_finetuned_seeded.py", [
 ('import sys, csv, random\n', 'import sys, csv, random, argparse\n'),
 ('    cfg = load_cfg()\n',
  '    ap = argparse.ArgumentParser()\n'
  '    ap.add_argument("--backbone", required=True)\n'
  '    ap.add_argument("--seed", type=int, required=True)\n'
  '    args = ap.parse_args()\n'
  '    cfg = load_cfg()\n'
  '    cfg["backbones"] = {k: v for k, v in cfg["backbones"].items() if k == args.backbone}\n'
  '    assert cfg["backbones"], f"unknown backbone {args.backbone}"\n'),
 ('    out_dir = STUDY / "outputs_finetuned"\n',
  '    out_dir = STUDY / "outputs_finetuned" / "seeds" / f"seed{args.seed}"\n'),
 ('        ckpt_dir = CKPT_DIR / bb\n',
  '        ckpt_dir = CKPT_DIR / "seeds" / f"seed{args.seed}" / bb\n'),
])
