# Gear Train Assembly
We will train two skills for the manipulator
- Peg Insertion
- Gear Meshing

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
  - Check if collision is working properly
- Complete MDP implementation and tune rewards
  - Add contact forces to observations
  - Refine reward functions
  - Define termination for task completion

