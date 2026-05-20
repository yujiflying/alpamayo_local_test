    t0_xyz = ego_history_xyz[-1].copy()
    t0_rot = spt.Rotation.from_matrix(ego_history_rot[-1])
    t0_rot_inv = t0_rot.inv()

    # ------------------------------------------------------------------
    # Step 1: construct the local frame directly from ego pose at t0
    #
    # This yields a local frame based on the offline ego pose convention.
    # ------------------------------------------------------------------
    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_matrix(ego_history_rot)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_matrix(ego_future_rot)).as_matrix()

    # ------------------------------------------------------------------
    # Step 2: convention fix for offline local frame
    #
    # Empirically, the offline local frame appears to differ from the model's
    # expected ego-local convention by a fixed +90° planar rotation.
    #
    # Apply:
    #   [x']   [ 0 -1  0 ] [x]
    #   [y'] = [ 1  0  0 ] [y]
    #   [z']   [ 0  0  1 ] [z]
    #
    # i.e.:
    #   x' = -y
    #   y' =  x
    #
    # This is a temporary alignment fix for offline data so that the local
    # frame better matches the model/action-space convention
    # (x=forward, y=lateral).
    # ------------------------------------------------------------------
    R_fix = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ],
        dtype=np.float64,
    )

    ego_history_xyz_local = ego_history_xyz_local @ R_fix.T
    ego_future_xyz_local = ego_future_xyz_local @ R_fix.T

    ego_history_rot_local = R_fix[None, :, :] @ ego_history_rot_local
    ego_future_rot_local = R_fix[None, :, :] @ ego_future_rot_local

    _debug_print(debug, f"t0_xyz={t0_xyz.tolist()}")
    _debug_print(debug, f"history_last_local_xyz={ego_history_xyz_local[-1].tolist()}")
    _debug_print(debug, f"future_first_local_xyz={ego_future_xyz_local[0].tolist()}")
    _debug_print(debug, f"future_last_local_xyz={ego_future_xyz_local[-1].tolist()}")

    history_idx = np.arange(num_history_steps, dtype=np.int64)
    future_idx = np.arange(num_future_steps, dtype=np.int64)

    data = {
        "ego_history_xyz": torch.from_numpy(ego_history_xyz_local[history_idx]).float().unsqueeze(0).unsqueeze(0),
        "ego_history_rot": torch.from_numpy(ego_history_rot_local[history_idx]).float().unsqueeze(0).unsqueeze(0),
        "ego_future_xyz": torch.from_numpy(ego_future_xyz_local[future_idx]).float().unsqueeze(0).unsqueeze(0),
        "ego_future_rot": torch.from_numpy(ego_future_rot_local[future_idx]).float().unsqueeze(0).unsqueeze(0),
        "image_frames": image_frames,
        "camera_indices": torch.from_numpy(np.asarray(camera_indices, dtype=np.int64)),
        "video_frame_indices": torch.from_numpy(video_frame_indices.astype(np.int64)).unsqueeze(0),
        "actual_video_frame_indices": torch.from_numpy(actual_video_frame_indices.astype(np.int64)).unsqueeze(0),
    }
    return data