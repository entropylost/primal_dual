// Hack to deal with nalgebra stack being slightly broken.
// TODO: File report.
#![allow(clippy::toplevel_ref_arg)]
#![allow(unused)]

use contact::Contact;
use macroquad::input::KeyCode;
use nalgebra::{self as na, matrix, stack, vector};
use std::fmt::Debug;
use std::{f32::consts::PI, ops::Deref};

mod opt;
mod split;
use split::Split;
mod contact;

type Vector3 = na::Vector3<f32>;
type Matrix3 = na::Matrix3<f32>;
type Matrix3x4 = na::Matrix3x4<f32>;
type Matrix4x3 = na::Matrix4x3<f32>;
type Matrix4 = na::Matrix4<f32>;
type Quaternion = na::Quaternion<f32>;
type UnitQuaternion = na::UnitQuaternion<f32>;

type Split3 = Split<Vector3, Vector3>;
type Position = Split<Vector3, UnitQuaternion>;
type Displacement = Split<Vector3, Quaternion>;
type Velocity = Split<Vector3, Vector3>;
type Force = Split<Vector3, Vector3>;
type Mass = Split<f32, Matrix3>;

impl Position {
    fn rotation_map(self) -> Matrix4x3 {
        let q = self.angular.quaternion().as_vector();
        1.0 / 2.0
            * matrix![
                q.w, q.z, -q.y;
                -q.z, q.w, q.x;
                q.y, -q.x, q.w;
                -q.x, -q.y, -q.z;
            ]
    }
    fn kinematic_map(self) -> Split<Matrix3, Matrix4x3> {
        Split::new(Matrix3::identity(), self.rotation_map())
    }
    fn map_velocity(self, velocity: Velocity) -> Displacement {
        Displacement {
            linear: velocity.linear,
            // Should be equal to the kinematic map times the velocity.
            angular: (Quaternion::from_imag(velocity.angular) * *self.angular) / 2.0,
        }
    }
    fn unconstrain(self) -> Displacement {
        Displacement {
            linear: self.linear,
            angular: *self.angular,
        }
    }
    fn step(self, velocity: Velocity) -> Self {
        (self.map_velocity(velocity) + self.unconstrain()).normalize()
    }
}
impl Displacement {
    fn normalize(self) -> Position {
        Position {
            linear: self.linear,
            angular: UnitQuaternion::from_quaternion(self.angular),
        }
    }
}

#[test]
fn test_rotation_map() {
    let pos = Position::from_angular(UnitQuaternion::from_quaternion(Quaternion::new(
        0.7, 3.0, 2.0, -1.0,
    )));
    let rot_map = pos.rotation_map();
    let vel = Vector3::new(7.0, -10.0, 2.6);
    assert_eq!(
        Quaternion::from_vector(rot_map * vel),
        pos.map_velocity(Velocity::from_angular(vel)).angular
    );
}

trait Constraint<const N: usize>: Debug {
    fn potential(&self, positions: [Position; N]) -> f32;
    fn force(&self, positions: [Position; N]) -> [Force; N];
    fn grad2_diag(&self, positions: [Position; N]) -> [Split3; N];
    // Add dual properties. Require indexing into a set of dual vectors or something.
}

#[derive(Debug, Clone, Copy)]
struct CosseratRod {
    radius: f32,
    young_modulus: f32,
    shear_modulus: f32,
    length: f32,
    rest_rotation: UnitQuaternion,
}
impl CosseratRod {
    fn resting_state(
        rod_radius: f32,
        young_modulus: f32,
        shear_modulus: f32,
        [pi, pj]: [Position; 2],
    ) -> Self {
        let length = (pj.linear - pi.linear).norm();
        let rest_rotation =
            UnitQuaternion::rotation_between(&Vector3::z(), &(pj.linear - pi.linear)).unwrap();
        Self {
            radius: rod_radius,
            young_modulus,
            shear_modulus,
            length,
            rest_rotation,
        }
    }
    fn stretch_shear_diag(self) -> Vector3 {
        let s = PI * self.radius.powi(2);
        let a = 5.0 / 6.0 * s;
        Vector3::new(
            self.shear_modulus * a,
            self.shear_modulus * a,
            self.young_modulus * s,
        )
    }
    fn stretch_shear(self) -> Matrix3 {
        Matrix3::from_diagonal(&self.stretch_shear_diag())
    }
    fn bend_twist_diag(self) -> Vector3 {
        let i = PI * self.radius.powi(4) / 4.0;
        let j = PI * self.radius.powi(4) / 2.0;
        Vector3::new(
            self.young_modulus * i,
            self.young_modulus * i,
            self.shear_modulus * j,
        )
    }
    fn bend_twist(self) -> Matrix3 {
        Matrix3::from_diagonal(&self.bend_twist_diag())
    }
    fn center_rotation(self, [pi, pj]: [Position; 2]) -> UnitQuaternion {
        pi.angular.nlerp(&pj.angular, 0.5)
    }
    fn center_rotation_gradient(self, [pi, pj]: [Position; 2]) -> Matrix4 {
        let qm = pi.angular.lerp(&pj.angular, 0.5);
        let qij = UnitQuaternion::from_quaternion(qm);
        (Matrix4::identity() - qij.as_vector() * qij.as_vector().transpose()) / qm.norm()
    }
}

#[derive(Debug, Clone, Copy)]
struct CosseratStretchShear {
    rod: CosseratRod,
}
impl Deref for CosseratStretchShear {
    type Target = CosseratRod;
    fn deref(&self) -> &Self::Target {
        &self.rod
    }
}

// p * q = rmul_mat(q) * p
fn rmul_mat(q: Quaternion) -> Matrix4 {
    stack![
        Matrix3::from_diagonal_element(q.scalar()) - cross_matrix(q.vector().into()), q.vector();
        -q.vector().transpose(), na::Matrix1::<f32>::new(q.scalar())
    ]
}

// p * q = lmul_mat(p) * q
fn lmul_mat(p: Quaternion) -> Matrix4 {
    stack![
        Matrix3::from_diagonal_element(p.scalar()) + cross_matrix(p.vector().into()), p.vector();
         -p.vector().transpose(), na::Matrix1::<f32>::new(p.scalar())
    ]
}

#[test]
fn test_quat_mul() {
    let p = Quaternion::new(1.0, 6.0, -2.0, 3.0);
    let q = Quaternion::new(7.0, -3.0, 5.0, 10.0);
    assert_eq!((p * q).coords, lmul_mat(p) * q.coords);
    assert_eq!((p * q).coords, rmul_mat(q) * p.coords);
}

// v.cross(w) = cross_matrix(v) * w
fn cross_matrix(v: Vector3) -> Matrix3 {
    matrix![
        0.0, -v.z, v.y;
        v.z, 0.0, -v.x;
        -v.y, v.x, 0.0
    ]
}

impl CosseratStretchShear {
    fn strain_measure(self, p @ [pi, pj]: [Position; 2]) -> Vector3 {
        1.0 / self.length
            * ((self.center_rotation(p) * self.rest_rotation).inverse() * (pj.linear - pi.linear))
            - Vector3::z()
    }
    // Wrt. the first position
    fn strain_gradient_lin(self, p: [Position; 2]) -> Matrix3 {
        -1.0 / self.length
            * (self.center_rotation(p) * self.rest_rotation)
                .to_rotation_matrix()
                .matrix()
                .transpose()
    }
    fn strain_gradient_ang(self, p @ [pi, pj]: [Position; 2]) -> Matrix3x4 {
        let qij = *self.center_rotation(p);
        let dqij = self.center_rotation_gradient(p);
        1.0 / self.length
            * self.rest_rotation.to_rotation_matrix().matrix().transpose()
            * rmul_mat(Quaternion::from_imag(pj.linear - pi.linear) * qij).fixed_view::<3, 4>(0, 0)
            * Matrix4::from_diagonal(&vector![-1.0, -1.0, -1.0, 1.0])
            * dqij
    }
}

impl Constraint<2> for CosseratStretchShear {
    fn potential(&self, p: [Position; 2]) -> f32 {
        let strain = self.strain_measure(p);
        (self.length / 2.0 * strain.transpose() * self.stretch_shear() * strain).to_scalar()
    }
    fn force(&self, p @ [pi, pj]: [Position; 2]) -> [Force; 2] {
        let strain = self.strain_measure(p);
        let force_i =
            -self.length * self.strain_gradient_lin(p).transpose() * self.stretch_shear() * strain;
        let force_j = -force_i;

        let torque_i =
            -self.length * self.strain_gradient_ang(p).transpose() * self.stretch_shear() * strain;
        let torque_j = torque_i; // Note: The torque isn't conserved but that's fine because its counterbalanced by the force.
        [
            Split::new(force_i, pi.rotation_map().transpose() * torque_i),
            Split::new(force_j, pj.rotation_map().transpose() * torque_j),
        ]
    }
    fn grad2_diag(&self, p @ [pi, pj]: [Position; 2]) -> [Split3; 2] {
        let lin = self.strain_gradient_lin(p);
        let ang = self.strain_gradient_ang(p);
        let grad_i = Split::new(lin, ang * pi.rotation_map());
        let diag_i = Split::new(
            grad_i.linear.transpose() * self.stretch_shear() * grad_i.linear,
            grad_i.angular.transpose() * self.stretch_shear() * grad_i.angular,
        );
        let diag_i = Split::new(diag_i.linear.diagonal(), diag_i.angular.diagonal());
        let grad_j = Split::new(-lin, -ang * pj.rotation_map());
        let diag_j = Split::new(
            grad_j.linear.transpose() * self.stretch_shear() * grad_j.linear,
            grad_j.angular.transpose() * self.stretch_shear() * grad_j.angular,
        );
        let diag_j = Split::new(diag_j.linear.diagonal(), diag_j.angular.diagonal());
        [diag_i, diag_j]
    }
}

#[derive(Debug, Clone, Copy)]
struct CosseratBendTwist {
    rod: CosseratRod,
}
impl Deref for CosseratBendTwist {
    type Target = CosseratRod;
    fn deref(&self) -> &Self::Target {
        &self.rod
    }
}

impl CosseratBendTwist {
    fn darboux_vector(self, p @ [pi, pj]: [Position; 2]) -> Vector3 {
        2.0 / self.length
            * (*self.center_rotation(p).conjugate() * (*pj.angular - *pi.angular)).imag()
    }
    fn darboux_gradient_ang(self, p @ [pi, pj]: [Position; 2]) -> Matrix3x4 {
        let dqij = self.center_rotation_gradient(p);
        let gradient = 2.0 / self.length
            * (1.0 / 2.0
                * rmul_mat(*pj.angular - *pi.angular)
                * Matrix4::from_diagonal(&vector![-1.0, -1.0, -1.0, 1.0])
                * dqij
                - lmul_mat(*self.center_rotation(p).conjugate()));
        gradient.fixed_view::<3, 4>(0, 0).into_owned()
    }
}

impl Constraint<2> for CosseratBendTwist {
    fn potential(&self, p: [Position; 2]) -> f32 {
        let darboux = self.darboux_vector(p);
        (self.length / 2.0 * darboux.transpose() * self.bend_twist() * darboux).to_scalar()
    }
    fn force(&self, p @ [pi, pj]: [Position; 2]) -> [Force; 2] {
        let darboux = self.darboux_vector(p);
        let torque_i =
            -self.length * self.darboux_gradient_ang(p).transpose() * self.bend_twist() * darboux;
        let torque_j = -self.length
            * self.darboux_gradient_ang([pj, pi]).transpose()
            * self.bend_twist()
            * -darboux;
        [
            Split::from_angular(pi.rotation_map().transpose() * torque_i),
            Split::from_angular(pj.rotation_map().transpose() * torque_j),
        ]
    }
    fn grad2_diag(&self, p @ [pi, pj]: [Position; 2]) -> [Split3; 2] {
        let grad_i = self.darboux_gradient_ang(p) * pi.rotation_map();
        let diag_i = grad_i.transpose() * self.bend_twist() * grad_i;
        let diag_i = Split::from_angular(diag_i.diagonal());
        let grad_j = self.darboux_gradient_ang([pj, pi]) * pj.rotation_map();
        let diag_j = grad_j.transpose() * self.bend_twist() * grad_j;
        let diag_j = Split::from_angular(diag_j.diagonal());
        [diag_i, diag_j]
    }
}

#[test]
fn test_cosserat() {
    let rod = CosseratRod {
        radius: 0.5,
        young_modulus: 1.0,
        shear_modulus: 1.0,
        length: 2.0,
        rest_rotation: UnitQuaternion::identity(),
    };
    let inv_rod = CosseratRod {
        radius: 0.5,
        young_modulus: 1.0,
        shear_modulus: 1.0,
        length: 2.0,
        rest_rotation: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI),
    };
    let se = CosseratStretchShear { rod };
    let inv_se = CosseratStretchShear { rod: inv_rod };
    let bt = CosseratBendTwist { rod };
    let pi = Split::from_linear(vector![0.0, -0.1, 0.0]);
    let pj = Split::from_linear(vector![2.0, 0.1, 0.0]);

    assert_eq!(bt.darboux_vector([pi, pj]), vector![0.0, 0.0, 0.0]);
    let diff = se.force([pi, pj])[0] - inv_se.force([pj, pi])[1];
    println!("{:?}", diff.angular);
    assert!(diff.linear.norm() < 1e-5);
    assert!(diff.angular.norm() < 1e-4);
}

#[test]
fn compare_cosserat_values() {
    fn assert_close(a: Split3, b: Split3) {
        assert!((a.linear - b.linear).norm() < 1e-6);
        assert!((a.angular - b.angular).norm() < 1e-6);
    }
    fn assert_close2(a: [Split3; 2], b: [Split3; 2]) {
        for i in 0..2 {
            assert!((a[i].linear - b[i].linear).norm() < 1e-6);
            assert!((a[i].angular - b[i].angular).norm() < 1e-6);
        }
    }

    let rod = CosseratRod {
        radius: 0.5,
        young_modulus: 1.24,
        shear_modulus: 2.6,
        length: 2.9,
        rest_rotation: UnitQuaternion::from_quaternion(Quaternion::new(-0.15, 0.2, -0.1, 1.0)),
    };
    let se = CosseratStretchShear { rod };
    let inv_se = CosseratStretchShear {
        rod: CosseratRod {
            rest_rotation: rod.rest_rotation
                * UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI),
            ..rod
        },
    };
    let bt = CosseratBendTwist { rod };

    let pi = Position {
        linear: vector![0.9, -0.3, 2.0],
        angular: UnitQuaternion::from_quaternion(Quaternion::new(1.0, -2.3, 4.0, 0.4)),
    };
    let pj = Position {
        linear: vector![2.0, 0.1, 0.0],
        angular: UnitQuaternion::from_quaternion(Quaternion::new(-0.6, 1.32, -0.4, -1.0)),
    };

    let potential = bt.potential([pi, pj]);
    assert_eq!(potential, 0.27819753);
    let force = bt.force([pi, pj]);
    assert_close2(
        force,
        [
            Split::from_angular(vector![-0.11077545, -0.2582397, -0.0065065296]),
            Split::from_angular(vector![0.11077541, 0.25823966, 0.006506523]),
        ],
    );
    let grad2 = bt.grad2_diag([pi, pj]);
    assert_close2(
        grad2,
        [Split::from_angular(vector![0.037722476, 0.12580885, 0.016810376]); 2],
    );
    // let (force_b, grad2_b) = opt::compute_bt_na(rod, [pi, pj]);
    // assert_close(force[0], force_b);
    // assert_close(grad2[0], grad2_b);

    let potential = se.potential([pi, pj]);
    assert_eq!(potential, 3.6220994);
    let force = se.force([pi, pj]);
    let rev_force = inv_se.force([pj, pi]);
    assert_close2(force, [rev_force[1], rev_force[0]]);
    assert_close2(
        force,
        [
            Split::new(
                vector![1.3247634, 0.61114, -1.0095686],
                vector![1.7519698, -0.14745957, -0.9239183],
            ),
            Split::new(
                vector![-1.3247634, -0.61114, 1.0095686],
                vector![-0.933517, -1.3915416, 1.066267],
            ),
        ],
    );
    let grad2 = se.grad2_diag([pi, pj]);
    assert_close2(
        grad2,
        [
            Split::new(
                vector![0.13895975, 0.18287909, 0.19864689],
                vector![0.6700756, 0.32354397, 0.5435535],
            ),
            Split::new(
                vector![0.13895975, 0.18287909, 0.19864689],
                vector![0.8507173, 0.43259573, 0.25385994],
            ),
        ],
    );
    // let (force_b, grad2_b) = opt::compute_se_na(rod, [pi, pj]);
    // assert_close(force[0], force_b);
    // assert_close(grad2[0], grad2_b);
}

#[derive(Debug)]
struct Constraint2 {
    targets: [usize; 2],
    constraint: Box<dyn Constraint<2>>,
}

#[macroquad::main("Pbd")]
async fn main() {
    let mut position: Vec<Position> = vec![vector![0.0, 0.0, 0.0], vector![2.0, 0.0, 0.0]]
        .into_iter()
        .map(Split::from_linear)
        .collect();
    let mut velocity: Vec<Velocity> = vec![
        Split::new(vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 0.1]),
        Split::new(vector![0.0, 0.0, 0.0], vector![0.0, 0.0, -0.1]),
    ];
    let mass: Vec<Mass> = vec![1.0, 1.0]
        .into_iter()
        .map(|x| Split::new(x, Matrix3::from_diagonal_element(2.0 / 5.0 * x * 0.5 * 0.5)))
        .collect();
    let particles = position.len();
    assert_eq!(particles, velocity.len());
    assert_eq!(particles, mass.len());

    let bt = CosseratBendTwist {
        rod: CosseratRod::resting_state(0.5, 1.0, 1.0, [position[0], position[1]]),
    };
    let se = CosseratStretchShear {
        rod: CosseratRod::resting_state(0.5, 1.0, 1.0, [position[0], position[1]]),
    };

    let constraints = [
        Constraint2 {
            targets: [0, 1],
            constraint: Box::new(se),
        },
        Constraint2 {
            targets: [0, 1],
            constraint: Box::new(bt),
        },
    ];

    let constraint_step = 0.1;

    loop {
        let running = macroquad::input::is_key_pressed(KeyCode::Period)
            || macroquad::input::is_key_down(KeyCode::Space);

        if running {
            let last_position = position.clone();
            let last_velocity = velocity.clone();
            for i in 0..particles {
                position[i] = position[i].step(velocity[i]);
            }

            let mut contacts: Vec<Constraint2> = vec![];

            for i in 0..particles {
                for j in i + 1..particles {
                    let pi = position[i];
                    let pj = position[j];
                    if (pi.linear - pj.linear).norm() <= 1.0 {
                        contacts.push(Constraint2 {
                            targets: [i, j],
                            constraint: Box::new(Contact {
                                normal: (pi.linear - pj.linear).normalize(),
                                stiffness: 10000.0,
                                length: 1.0,
                            }),
                        });
                    }
                }
            }
            for _iter in 0..1 {
                let mut forces = vec![Force::default(); particles];
                let mut grad2_diag = vec![Split::<Vector3, Vector3>::default(); particles];

                for Constraint2 {
                    targets,
                    constraint,
                } in constraints.iter().chain(&contacts)
                {
                    let [i, j] = *targets;
                    let pi = position[i];
                    let pj = position[j];
                    let p = [pi, pj];
                    let force = constraint.force(p);
                    let grad2 = constraint.grad2_diag(p);
                    forces[i] += force[0];
                    forces[j] += force[1];
                    grad2_diag[i] += grad2[0];
                    grad2_diag[j] += grad2[1];
                }
                let mut preconditioner_diag = vec![Split::<Vector3, Vector3>::default(); particles];
                for i in 0..particles {
                    preconditioner_diag[i].linear =
                        (Vector3::repeat(mass[i].linear) + grad2_diag[i].linear).map(|x| 1.0 / x);
                    preconditioner_diag[i].angular =
                        (mass[i].angular.diagonal() + grad2_diag[i].angular).map(|x| 1.0 / x);
                }
                let gradient = (0..particles)
                    .map(|i| mass[i] * (velocity[i] - last_velocity[i]) - forces[i])
                    .collect::<Vec<_>>();
                for i in 0..particles {
                    let step = Split {
                        linear: preconditioner_diag[i]
                            .linear
                            .component_mul(&gradient[i].linear),
                        angular: preconditioner_diag[i]
                            .angular
                            .component_mul(&gradient[i].angular),
                    };
                    velocity[i] -= constraint_step * step;
                }
                for i in 0..particles {
                    position[i] = last_position[i].step(velocity[i]);
                }
            }
        }
        {
            use macroquad::prelude::*;
            let scaling = 50.0;
            let offset = vector![200.0, 200.0];

            clear_background(BLACK);
            for p in &position {
                let pos = p.linear.xy() * scaling + offset;
                draw_circle(pos.x, pos.y, 0.5 * scaling, RED);
                let rot_x = (p.angular * vector![0.5, 0.0, 0.0]).xy() * scaling + pos;
                draw_line(pos.x, pos.y, rot_x.x, rot_x.y, 3.0, WHITE);
                let rot_y = (p.angular * vector![0.0, 0.5, 0.0]).xy() * scaling + pos;
                draw_line(pos.x, pos.y, rot_y.x, rot_y.y, 3.0, GREEN);
                let rot_z = (p.angular * vector![0.0, 0.0, 0.5]).xy() * scaling + pos;
                draw_line(pos.x, pos.y, rot_z.x, rot_z.y, 3.0, BLUE);
            }
            macroquad::window::next_frame().await
        }
    }
}
