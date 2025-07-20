import os
import subprocess

def run_command(cmd):
    print("\n" + "="*80)
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    print("="*80 + "\n")
    return result

def run_colmap_pipeline(image_dir, workspace_dir):
    os.makedirs(workspace_dir, exist_ok=True)
    db_path      = os.path.join(workspace_dir, "database.db")
    sparse_path  = os.path.join(workspace_dir, "sparse")
    dense_path   = os.path.join(workspace_dir, "dense")
    model_path   = os.path.join(dense_path, "fused.ply")

    # 1) Feature extraction (force PINHOLE intrinsics fallback)
    cmd_feat = (
        f"colmap feature_extractor "
        f"--database_path {db_path} "
        f"--image_path {image_dir} "
        f"--ImageReader.camera_model PINHOLE "
        f"--ImageReader.single_camera 1 "
        f"--ImageReader.default_focal_length_factor 1.5 "
        f"--SiftExtraction.use_gpu 0"
    )
    r = run_command(cmd_feat)
    if r.returncode != 0:
        raise RuntimeError("Feature extraction failed. Check your images/installation.")

    # 2) Exhaustive matching
    cmd_match = f"colmap exhaustive_matcher --database_path {db_path}"
    r = run_command(cmd_match)
    if r.returncode != 0:
        raise RuntimeError("Feature matching failed. Check your database file.")

    # 3) Incremental mapper (first attempt)
    os.makedirs(sparse_path, exist_ok=True)
    cmd_map = (
        f"colmap mapper "
        f"--database_path {db_path} "
        f"--image_path {image_dir} "
        f"--output_path {sparse_path} "
        # relax init inlier threshold
        f"--Mapper.init_min_num_inliers 15 "
        f"--Mapper.init_min_tri_angle 2.0"
    )
    r = run_command(cmd_map)
    if r.returncode != 0:
        print("❗ First mapper attempt failed, retrying with more relaxed settings...")
        # 4) Retry with even more relaxed settings
        cmd_map_relaxed = (
            cmd_map +
            " --Mapper.init_min_num_inliers 8 "
            " --Mapper.init_max_error 4.0"
        )
        r2 = run_command(cmd_map_relaxed)
        if r2.returncode != 0:
            raise RuntimeError(
                "Mapper failed twice. This usually means COLMAP couldn’t find a good initial image pair.\n"
                "→ Ensure your images have at least 30% overlap, are well-lit, sharp and from distinct angles."
            )

    # 5) Dense reconstruction
    os.makedirs(dense_path, exist_ok=True)
    run_command(
        f"colmap image_undistorter "
        f"--image_path {image_dir} "
        f"--input_path {sparse_path}/0 "
        f"--output_path {dense_path} "
        f"--output_type COLMAP"
    )
    run_command(f"colmap patch_match_stereo --workspace_path {dense_path} --workspace_format COLMAP")
    run_command(
        f"colmap stereo_fusion "
        f"--workspace_path {dense_path} "
        f"--workspace_format COLMAP "
        f"--output_path {model_path}"
    )

    if not os.path.exists(model_path):
        raise RuntimeError("Dense fusion failed to produce a model.")
    return model_path
