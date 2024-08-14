use std::{
    f32::consts::PI,
    ops::{Add, Deref, Mul, Neg, Sub},
};

use macroquad::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct Split<A, B> {
    linear: A,
    angular: B,
}
impl<A, B> Split<A, B> {
    fn new(linear: A, angular: B) -> Self {
        Self { linear, angular }
    }
}
impl<A, B: Default> Split<A, B> {
    fn from_linear(linear: A) -> Self {
        Self {
            linear,
            angular: Default::default(),
        }
    }
}
impl<A: Default, B> Split<A, B> {
    fn from_angular(angular: B) -> Self {
        Self {
            linear: Default::default(),
            angular,
        }
    }
}
impl<A, B> Neg for Split<A, B>
where
    A: Neg<Output = A>,
    B: Neg<Output = B>,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            linear: -self.linear,
            angular: -self.angular,
        }
    }
}
impl<A, B> Add<Self> for Split<A, B>
where
    A: Add<A, Output = A>,
    B: Add<B, Output = B>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            linear: self.linear + rhs.linear,
            angular: self.angular + rhs.angular,
        }
    }
}
impl<A, B> Sub<Self> for Split<A, B>
where
    A: Sub<A, Output = A>,
    B: Sub<B, Output = B>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            linear: self.linear - rhs.linear,
            angular: self.angular - rhs.angular,
        }
    }
}
impl<A, B, C, D> Mul<Split<C, D>> for Split<A, B>
where
    A: Mul<C>,
    B: Mul<D>,
{
    type Output = Split<A::Output, B::Output>;
    fn mul(self, rhs: Split<C, D>) -> Self::Output {
        Split {
            linear: self.linear * rhs.linear,
            angular: self.angular * rhs.angular,
        }
    }
}

type Position = Split<Vec3, Quat>;
type Displacement = Split<Vec3, Quat>;
type Velocity = Split<Vec3, Vec3>;
type Force = Split<Vec3, Vec3>;
type Mass = Split<f32, Mat3>;

impl Position {
    fn map_velocity(self, velocity: Velocity) -> Displacement {
        Displacement {
            linear: velocity.linear,
            angular: (Quat::from_vec4(velocity.angular.extend(0.0)) * self.angular) / 2.0,
        }
    }
}

trait Constraint<const N: usize> {
    fn potential(&self, positions: [Position; N]) -> f32;
    fn force(&self, positions: [Position; N]) -> [Force; N];
    fn diag_grad2(&self, positions: [Position; N]) -> [Split<Vec3, Vec3>; N];
}

#[derive(Debug, Clone, Copy)]
struct Rod {
    normal: Vec3,
    stiffness: f32,
    length: f32,
}
impl Constraint<2> for Rod {
    fn potential(&self, [pi, pj]: [Position; 2]) -> f32 {
        let value = (pi - pj).linear.dot(self.normal) - self.length;
        value * self.stiffness
    }
    fn force(&self, p: [Position; 2]) -> [Force; 2] {
        let value = self.potential(p);
        [
            -Split::from_linear(value * self.normal),
            Split::from_linear(value * self.normal),
        ]
    }
    fn diag_grad2(&self, _p: [Position; 2]) -> [Split<Vec3, Vec3>; 2] {
        [Split::from_linear(self.normal * self.normal * self.stiffness); 2]
    }
}

#[derive(Debug, Clone, Copy)]
struct Contact {
    normal: Vec3,
    stiffness: f32,
    length: f32,
}
impl Constraint<2> for Contact {
    fn potential(&self, p @ [pi, pj]: [Position; 2]) -> f32 {
        let value = (pi - pj).linear.dot(self.normal) - self.length;
        (value * self.stiffness).min(0.0)
    }
    fn force(&self, p: [Position; 2]) -> [Force; 2] {
        let value = self.potential(p);
        [
            -Split::from_linear(value * self.normal),
            Split::from_linear(value * self.normal),
        ]
    }
    fn diag_grad2(&self, p: [Position; 2]) -> [Split<Vec3, Vec3>; 2] {
        if self.potential(p) < 0.0 {
            [Split::from_linear(self.normal * self.normal * self.stiffness); 2]
        } else {
            [Split::default(); 2]
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RadialContact {
    stiffness: f32,
    length: f32,
}
impl Constraint<2> for RadialContact {
    fn potential(&self, [pi, pj]: [Position; 2]) -> f32 {
        let value = (pi - pj).linear.length() - self.length;
        (value * self.stiffness).min(0.0)
    }
    fn force(&self, p @ [pi, pj]: [Position; 2]) -> [Force; 2] {
        let normal = (pi - pj).linear.normalize();
        let value = self.potential(p);
        [
            -Split::from_linear(value * normal),
            Split::from_linear(value * normal),
        ]
    }
    fn diag_grad2(&self, p @ [pi, pj]: [Position; 2]) -> [Split<Vec3, Vec3>; 2] {
        let normal = (pi - pj).linear.normalize();

        if self.potential(p) < 0.0 {
            [Split::from_linear(normal * normal * self.stiffness); 2]
        } else {
            [Split::default(); 2]
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CosseratParameters {
    rod_radius: f32,
    young_modulus: f32,
    shear_modulus: f32,
    length: f32,
    rest_rotation: Quat,
}
impl CosseratParameters {
    fn stretch_shear_diag(self) -> Vec3 {
        let s = PI * self.rod_radius.powi(2);
        let a = 5.0 / 6.0 * s;
        Vec3::new(
            self.shear_modulus * a,
            self.shear_modulus * a,
            self.young_modulus * s,
        )
    }
    fn bend_twist_diag(self) -> Vec3 {
        let i = PI * self.rod_radius.powi(4) / 4;
        let j = PI * self.rod_radius.powi(4) / 2;
        Vec3::new(
            self.young_modulus * i,
            self.young_modulus * i,
            self.shear_modulus * j,
        )
    }
    fn center_rotation(self, [pi, pj]: [Position; 2]) -> Quat {
        pi.angular.lerp(pj.angular, 0.5)
    }
}

#[derive(Debug, Clone, Copy)]
struct CosseratStretchShear {
    params: CosseratParameters,
}
impl Deref for CosseratStretchShear {
    type Target = CosseratParameters;
    fn deref(&self) -> &Self::Target {
        &self.params
    }
}
impl CosseratStretchShear {
    fn strain_measure(self, p @ [pi, pj]: [Position; 2]) -> Vec3 {
        1.0 / self.length
            * ((self.center_rotation(p) * self.rest_rotation).inverse() * (pj - pi).linear)
            - Vec3::Z
    }
    // Wrt. the first position
    // Also, this is the jacobian actually, so no transpose is needed in the later part.
    fn strain_gradient_lin(self, p @ [pi, pj]: [Position; 2]) -> Mat3 {
        -1.0 / self.length * Mat3::from_quat(self.center_rotation(p) * self.rest_rotation)
    }
}

impl Constraint<2> for CosseratStretchShear {
    fn potential(&self, p: [Position; 2]) -> f32 {
        let strain = self.strain_measure(p);
        (self.length / 2.0 * strain * self.stretch_shear_diag() * strain).element_sum()
    }
    fn force(&self, p: [Position; 2]) -> [Force; 2] {
        let strain = self.strain_measure(p);
        let force = -self.length * self.strain_gradient_lin(p) * (self.stretch_shear_diag * strain);
        let torque = Vec3::ZERO; // TODO: Finish.
        let force = Split::new(force, torque);
        [force, -force]
    }
    fn diag_grad2(&self, positions: [Position; 2]) -> [Split<Vec3, Vec3>; 2] {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct Particles {
    position: Vec<Position>,
    last_position: Vec<Position>,
    velocity: Vec<Velocity>,
    last_velocity: Vec<Velocity>,
    mass: Vec<Mass>,
}
impl Particles {
    fn len(&self) -> usize {
        self.position.len()
    }
}

#[macroquad::main("main")]
async fn main() {
    let scaling = 50.0;

    let mut particles = Particles {
        position: vec![
            vec2(0.0, 0.0),
            vec2(4.0, 0.0),
            vec2(6.0, 0.0),
            vec2(7.0, 0.0),
        ],
        last_position: vec![vec2(0.0, 0.0); 4],
        velocity: vec![
            vec2(0.1, 0.0),
            vec2(0.0, 0.0),
            vec2(0.0, 0.0),
            vec2(0.0, 0.0),
        ],
        last_velocity: vec![vec2(0.0, 0.0); 4],
        mass: vec![1.0, 1.0, 1.0, 1.0],
    };

    let constraint_step = 0.1;

    loop {
        let offset = vec2(100.0, 100.0);
        clear_background(BLACK);

        if is_key_pressed(KeyCode::Period) || is_key_down(KeyCode::Space) {
            particles.last_position = particles.position.clone();
            particles.last_velocity = particles.velocity.clone();
            for i in 0..particles.len() {
                particles.position[i] += particles.velocity[i];
            }

            let mut contacts = vec![];

            for i in 0..particles.len() {
                for j in i + 1..particles.len() {
                    let pi = particles.position[i];
                    let pj = particles.position[j];
                    if pi.distance(pj) <= 1.0 {
                        contacts.push(Contact {
                            colliders: (i, j),
                            normal: (pi - pj).normalize(),
                            stiffness: 10000.0,
                        });
                    }
                }
            }
            for _iter in 0..100 {
                let mut forces = vec![Vec2::ZERO; particles.len()];
                let mut jacobian_diag = vec![Vec2::ZERO; particles.len()];

                for &Contact {
                    colliders: (i, j),
                    normal,
                    stiffness,
                } in &contacts
                {
                    let pi = particles.position[i];
                    let pj = particles.position[j];
                    let constraint = (pi - pj).dot(normal) - 1.0;
                    let force = normal * stiffness * constraint.min(0.0);
                    forces[i] -= force;
                    forces[j] += force;
                    jacobian_diag[i] += normal * normal * stiffness;
                    jacobian_diag[j] += normal * normal * stiffness;
                }
                let mut preconditioner_diag = vec![Vec2::ZERO; particles.len()];
                for i in 0..particles.len() {
                    preconditioner_diag[i] = 1.0 / (particles.mass[i] + jacobian_diag[i])
                }
                let gradient = (0..particles.len())
                    .map(|i| {
                        particles.mass[i] * (particles.velocity[i] - particles.last_velocity[i])
                            - forces[i]
                    })
                    .collect::<Vec<_>>();
                for i in 0..particles.len() {
                    particles.velocity[i] -= constraint_step * preconditioner_diag[i] * gradient[i];
                }
                for i in 0..particles.len() {
                    particles.position[i] = particles.velocity[i] + particles.last_position[i];
                }
            }
        }
        for i in 0..particles.len() {
            draw_circle(
                particles.position[i].x * scaling + offset.x,
                particles.position[i].y * scaling + offset.x,
                0.5 * scaling,
                RED,
            );
        }
        next_frame().await
    }
}
