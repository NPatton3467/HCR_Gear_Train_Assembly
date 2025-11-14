# Gear Train Assembly
We will train two skills for the manipulator
- Peg Insertion
- Gear Meshing

## Pre-Requisite
- Isaac Lab [https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html]
- Download custom assets, i.e. the custom robot and the assembly board
  - https://utexas.box.com/s/burh5f9f1n102pabi0wzztsxd1sbt46o
  - Save them to `~/Documents/USD/`

## Setup
1. Go to the directory `source/isaaclab_tasks/isaaclab_tasks/manager_based`
2. Clone the repository `https://github.com/bugartlan/HCR_Gear_Train_Assembly.git`

## Hardware
- UR3e
- Robotiq Hand-E

## Peg Insertion
### Current Progress
- Scene is set up with the gripper already holding the peg.
- The hole is fixed on the table.
- The bare-bones of the MDP are implemented.

### TODOs
- Configure the parameters for the assets, e.g. peg and hole
  - Dimensions of the assets
- Complete MDP implementation and tune rewards
  - Add contact forces to observations
  - Refine reward functions
  - Define termination for task completion
- Resolve potential interference between the fingers and the gears

### Training 
- **Run training without visualization (headless)**: Use `--num_env 1` for faster startup for quick syntax checks.
```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Assembly-PegInsert-v0 --headless
```
- **Run training with visualization** for visual debugging.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Assembly-PegInsert-v0 --num_env 1
```

### Playing a Policy
- **Export and run the most recent trained policy**:
```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Assembly-PegInsert-v0 --num_env 1
```

