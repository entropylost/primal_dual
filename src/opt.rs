use std::f32::consts::PI;

use macroquad::math::{Mat3, Mat4, Vec3, Vec4, Vec4Swizzles};

use crate::{split::Split, CosseratRod, Position, Split3};

type Spl3 = Split<Vec3, Vec3>;

pub fn compute_se_na(rod: CosseratRod, [pi, pj]: [Position; 2]) -> (Split3, Split3) {
    let (a, b) = compute_se(
        rod.radius,
        rod.young_modulus,
        rod.shear_modulus,
        rod.length,
        rod.rest_rotation.coords.into(),
        [pi.linear.into(), pj.linear.into()],
        [pi.angular.coords.into(), pj.angular.coords.into()],
    );
    (
        Split3::new(a.linear.into(), a.angular.into()),
        Split3::new(b.linear.into(), b.angular.into()),
    )
}

fn compute_se(
    radius: f32,
    young_modulus: f32,
    shear_modulus: f32,
    length: f32,
    rest_rotation: Vec4,
    [pi, pj]: [Vec3; 2],
    [qi, qj]: [Vec4; 2],
) -> (Spl3, Spl3) {
    todo!()
}

pub fn compute_bt_na(rod: CosseratRod, [pi, pj]: [Position; 2]) -> (Split3, Split3) {
    let (a, b) = compute_bt(
        rod.radius,
        rod.young_modulus,
        rod.shear_modulus,
        rod.length,
        rod.rest_rotation.coords.into(),
        [pi.angular.coords.into(), pj.angular.coords.into()],
    );
    (
        Split3::from_angular(a.into()),
        Split3::from_angular(b.into()),
    )
}

fn conjugate(q: Vec4) -> Vec4 {
    q * Vec3::splat(-1.0).extend(1.0)
}

// Extracts the imaginary part of the p * q quaternion product
fn qmul_imag(p: Vec4, q: Vec4) -> Vec3 {
    p.w * q.xyz() + q.w * p.xyz() + p.xyz().cross(q.xyz())
}

fn compute_bt(
    radius: f32,
    young_modulus: f32,
    shear_modulus: f32,
    length: f32,
    rest_rotation: Vec4,
    [qi, qj]: [Vec4; 2],
) -> (Vec3, Vec3) {
    // Constants
    let bend_twist_coeff = {
        let i = PI * radius.powi(4) / 4.0;
        let j = PI * radius.powi(4) / 2.0;
        Vec3::new(young_modulus * i, young_modulus * i, shear_modulus * j)
    };

    let qm = (qi + qj) / 2.0;
    let qm_norm = qm.length();
    let qij = qm / qm_norm;
    let qij_c = conjugate(qij);
    let qdiff = qj - qi;

    let darboux = 2.0 / length * qmul_imag(qij_c, qdiff);

    todo!()
}
