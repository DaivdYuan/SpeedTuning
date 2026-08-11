# Experiment manifests

`paper_sim.json` is the best simulation configuration recoverable from the
published paper and the retained `SpeedTuningViz` experiment directory names. It
is an archival preset, not a guarantee of exact numerical reproduction.

`scripted_tea_bag.json` is the fully runnable public reproduction. It uses the
included tea-bag waypoint policy and the latest retained training recipe from
the repository history: five speed actions, state observations, quadratic speed
reward, 100k decisions, and episode-boundary Rainbow updates. Load it with
`--config scripted-tea-bag`.

`scripted_tea_bag_randomized.json` inherits that recipe and samples the tea bag
within the original ACT cube-position range at every reset. This opt-in variant
is the appropriate preset for reporting success rates across simulator seeds;
the exact historical tea-bag environment uses one fixed initial pose.

`scripted_pick_and_place.json` and `scripted_insertion.json` apply the same
scripted-base Rainbow recipe to the other reconstructed tasks. Pick-and-place
includes speed actions through 4.5x because its transport phase tolerates the
paper's higher acceleration range; insertion retains the denser historical
1.0x-3.0x actions for contact-sensitive control.

The main recovered choices are:

- discrete speeds: 1.5, 2.0, 3.0, and 4.5;
- frame skip 10 and observation stack 5;
- proprioception plus top-camera features from pretrained ResNet-18;
- no privileged simulator object state;
- quadratic speed reward and hidden dimension 1024.

The exact original task-policy checkpoints, speed-policy checkpoints, datasets,
and full training command were not recovered. Values retained only in the old
trainer, such as gamma and replay capacity, are included to make the preset
concrete and are identified by the manifest's archival status.

Files in `ablations/` inherit the paper preset and override one experimental
factor. They can be passed directly to `--config`; command-line arguments may
override any value.
