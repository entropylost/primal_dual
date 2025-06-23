use either::Either;

use super::*;

pub enum Solvers {
    Primal(PrimalSolver),
    ScaledPrimal(ScaledPrimalSolver),
    Dual(DualSolver),
}
impl Solvers {
    pub fn name(&self) -> &'static str {
        match self {
            Solvers::Primal(_) => "Primal",
            Solvers::ScaledPrimal(_) => "Scaled Primal",
            Solvers::Dual(_) => "Dual",
        }
    }
    pub fn iterations(&self) -> usize {
        match self {
            Solvers::Primal(solver) => solver.iterations,
            Solvers::ScaledPrimal(solver) => solver.iterations,
            Solvers::Dual(solver) => solver.iterations,
        }
    }
    pub fn constraint_step(&self) -> Real {
        match self {
            Solvers::Primal(solver) => solver.constraint_step,
            Solvers::ScaledPrimal(solver) => solver.constraint_step,
            Solvers::Dual(solver) => solver.constraint_step,
        }
    }
    pub fn diag_precond(&self) -> bool {
        match self {
            Solvers::Primal(solver) => solver.diag_precond,
            Solvers::ScaledPrimal(_) => true,
            Solvers::Dual(solver) => solver.diag_precond,
        }
    }
}
impl Solver for Solvers {
    fn solve(
        &mut self,
        mass: &[Mass],
        constraints: &[ConstraintBox],
        last_position: &[Position],
        last_velocity: &[Velocity],
        position: &mut [Position],
        velocity: &mut [Velocity],
    ) {
        match self {
            Solvers::Primal(solver) => solver.solve(
                mass,
                constraints,
                last_position,
                last_velocity,
                position,
                velocity,
            ),
            Solvers::ScaledPrimal(solver) => solver.solve(
                mass,
                constraints,
                last_position,
                last_velocity,
                position,
                velocity,
            ),
            Solvers::Dual(solver) => solver.solve(
                mass,
                constraints,
                last_position,
                last_velocity,
                position,
                velocity,
            ),
        }
    }
}

pub trait Solver {
    fn solve(
        &mut self,
        mass: &[Mass],
        constraints: &[ConstraintBox],
        last_position: &[Position],
        last_velocity: &[Velocity],
        position: &mut [Position],
        velocity: &mut [Velocity],
    );
}

pub struct PrimalSolver {
    pub iterations: usize,
    pub constraint_step: Real,
    pub diag_precond: bool,
}

impl Solver for PrimalSolver {
    fn solve(
        &mut self,
        mass: &[Mass],
        constraints: &[ConstraintBox],
        last_position: &[Position],
        last_velocity: &[Velocity],
        position: &mut [Position],
        velocity: &mut [Velocity],
    ) {
        let particles = mass.len();
        for _ in 0..self.iterations {
            let mut forces = vec![Force::default(); particles];
            let mut hessians = if self.diag_precond {
                Either::Left(
                    mass.iter()
                        .map(|m| Split::new(Vector::repeat(m.linear), m.angular.diagonal()))
                        .collect::<Vec<_>>(),
                )
            } else {
                Either::Right(
                    mass.iter()
                        .map(|m| {
                            Split::new(
                                MatrixP::identity() * m.linear,
                                MatrixW::identity() * m.angular,
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            };

            for ConstraintBox {
                targets,
                constraint,
            } in constraints
            {
                let p = targets.iter().map(|&i| position[i]).collect::<Vec<_>>();

                let force = constraint.force(&p);
                for (i, &j) in targets.iter().enumerate() {
                    forces[j] += force[i];
                }

                match &mut hessians {
                    Either::Left(hessians) => {
                        let hessian = constraint.hessian_diag(&p);
                        for (i, &j) in targets.iter().enumerate() {
                            hessians[j] += hessian[i];
                        }
                    }
                    Either::Right(hessians) => {
                        let hessian = constraint.hessian(&p);
                        for (i, &j) in targets.iter().enumerate() {
                            hessians[j] += hessian[i];
                        }
                    }
                }
            }
            let step = (0..particles)
                .map(|i| {
                    let grad = mass[i] * (velocity[i] - last_velocity[i]) - forces[i];
                    match &hessians {
                        Either::Left(hessians) => {
                            let precond = hessians[i].reciprocal();
                            precond.component_mul(grad)
                        }
                        Either::Right(hessians) => {
                            let precond = hessians[i].inverse();
                            precond * grad
                        }
                    }
                })
                .collect::<Vec<_>>();
            for (i, step) in step.into_iter().enumerate() {
                if mass[i].linear.is_infinite() || mass[i].angular.iter().any(|x| x.is_infinite()) {
                    continue;
                }
                velocity[i] -= self.constraint_step * step;
            }
            for i in 0..particles {
                position[i] = last_position[i].step(velocity[i]);
            }
        }
    }
}

pub struct DualSolver {
    pub iterations: usize,
    pub constraint_step: Real,
    pub diag_precond: bool,
}
impl Solver for DualSolver {
    fn solve(
        &mut self,
        mass: &[Mass],
        constraints: &[ConstraintBox],
        last_position: &[Position],
        _last_velocity: &[Velocity],
        position: &mut [Position],
        velocity: &mut [Velocity],
    ) {
        let particles = mass.len();
        let mut dual_vars = constraints
            .iter()
            .map(|x| DVector::zeros(x.constraint.dim_v()))
            .collect::<Vec<_>>();
        for _ in 0..self.iterations {
            for (
                ConstraintBox {
                    targets,
                    constraint,
                },
                dual_var,
            ) in constraints.iter().zip(dual_vars.iter_mut())
            {
                let p = targets.iter().map(|&i| position[i]).collect::<Vec<_>>();
                let m = targets.iter().map(|&i| mass[i]).collect::<Vec<_>>();
                let dual_force =
                    -constraint.value(&p) - dual_var.component_div(&constraint.stiffness());
                let precond = if self.diag_precond {
                    DMatrix::from_diagonal(&constraint.dual_preconditioner_diag(&p, &m))
                } else {
                    constraint.dual_preconditioner(&p, &m)
                };
                let delta = self.constraint_step * precond * dual_force;
                *dual_var += &delta;
                let jacobian = constraint.jacobian(&p);
                for (j, &k) in targets.iter().enumerate() {
                    velocity[k].linear +=
                        mass[k].linear.inverse() * jacobian[j].linear.transpose() * &delta;
                    velocity[k].angular +=
                        mass[k].angular.inverse() * jacobian[j].angular.transpose() * &delta;
                }
            }
            for i in 0..particles {
                position[i] = last_position[i].step(velocity[i]);
            }
        }
    }
}

pub struct ScaledPrimalSolver {
    pub iterations: usize,
    pub constraint_step: Real,
    pub starting_stiffness: Option<Real>,
    pub stiffness_scaling: Option<Real>,
}
impl Solver for ScaledPrimalSolver {
    fn solve(
        &mut self,
        mass: &[Mass],
        constraints: &[ConstraintBox],
        last_position: &[Position],
        last_velocity: &[Velocity],
        position: &mut [Position],
        velocity: &mut [Velocity],
    ) {
        let particles = mass.len();

        let min_stiffness = constraints
            .iter()
            .map(|x| x.constraint.stiffness().min())
            .reduce(Real::min)
            .unwrap_or(1.0);
        let max_stiffness = constraints
            .iter()
            .map(|x| x.constraint.stiffness().max())
            .reduce(Real::max)
            .unwrap_or(1.0);
        let starting_stiffness = self.starting_stiffness.unwrap_or(min_stiffness);
        let stiffness_scaling = self.stiffness_scaling.unwrap_or(
            (max_stiffness / starting_stiffness).powf(((self.iterations - 1) as Real).recip()),
        );

        for iter in 0..self.iterations {
            let max_stiffness = starting_stiffness * stiffness_scaling.powi(iter as i32);

            let mut forces = vec![Force::default(); particles];
            let mut hessians = mass
                .iter()
                .map(|m| Split::new(Vector::repeat(m.linear), m.angular.diagonal()))
                .collect::<Vec<_>>();

            for ConstraintBox {
                targets,
                constraint,
            } in constraints
            {
                let mut constraint = constraint.clone();
                let stiffness = constraint.stiffness().map(|x| x.min(max_stiffness));
                constraint.stiffness_mut().copy_from(&stiffness);
                let p = targets.iter().map(|&i| position[i]).collect::<Vec<_>>();

                let force = constraint.force(&p);
                for (i, &j) in targets.iter().enumerate() {
                    forces[j] += force[i];
                }

                let hessian = constraint.hessian_diag(&p);
                for (i, &j) in targets.iter().enumerate() {
                    hessians[j] += hessian[i];
                }
            }
            let step = (0..particles)
                .map(|i| {
                    let grad = mass[i] * (velocity[i] - last_velocity[i]) - forces[i];
                    let precond = hessians[i].reciprocal();
                    precond.component_mul(grad)
                })
                .collect::<Vec<_>>();
            for (i, step) in step.into_iter().enumerate() {
                if mass[i].linear.is_infinite() || mass[i].angular.iter().any(|x| x.is_infinite()) {
                    continue;
                }
                velocity[i] -= self.constraint_step * step;
            }
            for i in 0..particles {
                position[i] = last_position[i].step(velocity[i]);
            }
        }
    }
}
