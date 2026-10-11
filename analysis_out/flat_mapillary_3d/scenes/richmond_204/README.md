# richmond:204

Frame: the SfM model frame, metric and approximately east-north-up (x east, y north, z up, metres) about the corner origin, the source click raycast at 2.6 m. Camera poses in cameras.json are camera-to-world (OpenCV axes). points.json holds the GT click in 3D per lift, and per pair each arm's prediction and the reference as rays from the other camera. Scene files listed in cameras.json -> files; those not committed are on makelab2 with their sha256.
