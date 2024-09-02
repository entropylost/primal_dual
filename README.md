# An Implementation of "Primal/Dual Descent Methods for Dynamics"

This repository contains an implementation of primal and dual solvers in Rust for rigid body physics from [the paper](https://mmacklin.com/primaldual.pdf), in an easily-extensible manner. It also has a formulation of Cosserat rods implemented using this, taken from "Rod-Bonded Discrete Element Method" by Zhang et al.

To run, first install Cargo, and then execute `cargo r`. Note that this program is not at all optimized, and a lot of the calculations can be sped up.
