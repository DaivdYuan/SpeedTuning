# Changelog

## 0.1.0

- Released pick-and-place, insertion, and tea-bag MuJoCo tasks.
- Added parameterized execution speed for retained waypoint policies.
- Added a model-agnostic variable-speed action-chunk interface.
- Added external chunk-policy and speed-policy factory loading.
- Added supported Rainbow DQN speed-policy training, checkpoints, and evaluation.
- Added decision-level speed execution with fresh receding-horizon chunks and
  shared frame-skip semantics for training and evaluation.
- Added stacked proprioceptive/visual speed observations with pretrained, random,
  and external image-encoder support.
- Added retained ACT checkpoint/backbone adapters, checkpointed preprocessing,
  seeded physical-acceleration metrics, fixed-speed sweeps, and plotting.
- Added an archival paper configuration and manifests for every published
  simulation ablation.
- Added runnable scripted-policy presets for pick-and-place, insertion, and tea
  bag, including the retained reward and Rainbow update schedule.
- Added from-scratch reproduction instructions and machine-readable reference
  results for all three simulated tasks.
- Added seeded tea-bag pose randomization as a separately labeled robustness
  protocol while preserving the fixed-pose historical environment.
- Added periodic training snapshots, task/protocol metadata validation, and safe
  loading for locally generated speed-policy checkpoints.
- Added clean-install packaging, continuous integration, and release tests.
- Removed real-robot, private-path, scratch-output, and trained-checkpoint
  artifacts from the public surface.
