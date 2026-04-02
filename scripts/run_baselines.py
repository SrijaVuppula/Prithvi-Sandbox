def main(cfg: dict):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = cfg["data"]["patch_size"]
    output_dir = Path(cfg["experiment"]["output_dir"])
    save_plots = cfg["experiment"].get("save_plots", True)
    plot_limit = cfg["experiment"].get("save_plots_limit", 10)
    plot_count = 0

    logger = ExperimentLogger(output_dir=str(output_dir))

    # resume support
    completed_runs = set()
    if logger.csv_path.exists():
        import csv
        with open(logger.csv_path) as f:
            for row in csv.DictReader(f):
                completed_runs.add((
                    row["tile_id"], row["backbone"], row["mask_position"],
                    row["n_frames"], row["gap_type"]
                ))
        if completed_runs:
            print(f"Resuming — {len(completed_runs)} conditions already done.")

    tiles       = cfg["data"]["tiles"]
    backbones   = cfg["model"]["backbones"]
    positions   = cfg["experiment"]["mask_positions"]
    seq_lengths = cfg["experiment"]["sequence_lengths"]
    gap_types   = cfg["experiment"]["gap_types"]

    total = (len(tiles) * len(backbones) * len(positions)
             * len(seq_lengths) * len(gap_types))
    done  = 0

    for tile_cfg in tiles:
        tile_id      = tile_cfg["id"]
        tile_base    = Path(tile_cfg["base_dir"])
        all_files    = tile_cfg["files"]

        for bb, pos, n_frames, gap_cfg in product(
            backbones, positions, seq_lengths, gap_types
        ):
            done += 1
            gap_name  = gap_cfg["name"]
            bb_base   = Path(bb["base_dir"])

            # resume check
            condition_key = (tile_id, bb["name"], pos, str(n_frames), gap_name)
            if condition_key in completed_runs:
                print(f"[{done}/{total}] SKIP (done): {condition_key}")
                continue

            print(f"\n[{done}/{total}] backbone={bb['name']} pos={pos} "
                  f"T={n_frames} gap={gap_name} tile={tile_id}")

            # patch this backbone's source files
            apply_all_patches(bb_base)

            # select files for this condition
            try:
                selected_files = select_files_for_condition(
                    all_files, n_frames, gap_cfg
                )
            except ValueError as e:
                print(f"  SKIP: {e}")
                continue

            gap_days = compute_gap_days(selected_files)

            # load model
            try:
                model, bands, mean, std = load_model(
                    base_dir=bb_base,
                    checkpoint_filename=bb["checkpoint"],
                    num_frames=n_frames,
                    device=device,
                )
            except Exception as e:
                print(f"  ERROR loading model: {e}")
                continue

            # load data
            try:
                x, temporal_coords, location_coords, _ = \
                    load_sample_from_inference_module(
                        base_dir=tile_base,
                        file_paths=selected_files,
                        mean=mean,
                        std=std,
                        device=device,
                    )
            except Exception as e:
                print(f"  ERROR loading data: {e}")
                del model
                torch.cuda.empty_cache()
                continue

            # run inference
            frame_idx = get_masked_frame_index(pos, n_frames)
            try:
                result = run_one_condition(
                    model=model,
                    x=x,
                    temporal_coords=temporal_coords,
                    location_coords=location_coords,
                    frame_idx=frame_idx,
                    patch_size=patch_size,
                    device=device,
                )
            except Exception as e:
                print(f"  ERROR during inference: {e}")
                del model
                torch.cuda.empty_cache()
                continue

            # metrics
            pred_frame = result["rec_img"][0, :, frame_idx]
            gt_frame   = result["x_cpu"][0, :, frame_idx]
            metrics    = evaluate_reconstruction(pred_frame, gt_frame, mean, std)

            # log
            run_id = logger.log(
                backbone=bb["name"],
                mask_position=pos,
                n_frames=n_frames,
                gap_type=gap_name,
                tile_id=tile_id,
                masked_frame_idx=frame_idx,
                gap_days=gap_days,
                metrics=metrics,
                loss=result["loss"],
                mask_ratio=result["mask_ratio"],
                checkpoint=bb["checkpoint"],
            )

            # plot
            if save_plots and plot_count < plot_limit:
                plot_path = output_dir / "plots" / f"{run_id}.png"
                save_plot(plot_path, result["x_cpu"], result["rec_img"],
                          frame_idx, bands, mean, std)
                plot_count += 1

            del model
            torch.cuda.empty_cache()

    print(f"\nSweep complete. {done} conditions run.")
    print(f"Results: {output_dir / 'results.csv'}")