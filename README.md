# An Implementation of "Primal/Dual Descent Methods for Dynamics"

This repository contains a comparison of primal and dual solvers in Rust for rigid body physics from [the paper](https://mmacklin.com/primaldual.pdf), as well as an [augmented lagrangian](https://en.wikipedia.org/wiki/Augmented_Lagrangian_method) ("scaled primal") solver taken from the finite-stiffness formulation of [AVBD](https://graphics.cs.utah.edu/research/projects/avbd/). It also has a formulation of Cosserat rods implemented with these solvers, taken from "Rod-Bonded Discrete Element Method" by Zhang et al.

To run, first install Cargo, and then execute `cargo r`. Note that this program is not at all optimized, and a lot of the calculations can be sped up.

Controls:
* `[SPACE]`: Pause / Unpause
* `.`: Step forward one frame
* `[ESC]`: Exit
