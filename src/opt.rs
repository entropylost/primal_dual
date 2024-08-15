use std::{env::current_exe, f32::consts::PI};

use crate::{split::Split, CosseratRod, Matrix3x4, Matrix4x3, Position, Split3};

use luisa::lang::types::vector as lv;
use luisa::prelude::*;
use luisa_compute::{self as luisa, lang::autodiff::grad, DeviceType};
use lv::{Mat3, Mat4};
use nalgebra::{Matrix3, Vector3, Vector4};

type Vec3 = lv::Vec3<f32>;
type Vec4 = lv::Vec4<f32>;

#[repr(C)]
#[derive(Debug, Clone, Copy, Value, PartialEq)]
struct Outputs {
    se_lin_force: Vec3,
    se_ang_force: Vec3,
    se_lin_grad2: Vec3,
    se_ang_grad2: Vec3,
    bt_ang_force: Vec3,
    bt_ang_grad2: Vec3,
}

pub fn compute_na(rod: CosseratRod, [pi, pj]: [Position; 2]) -> (Split3, Split3, Split3, Split3) {
    let outputs = compute_host(
        rod.radius,
        rod.young_modulus,
        rod.shear_modulus,
        rod.length,
        rod.rest_rotation.coords.into(),
        [pi.linear.into(), pj.linear.into()],
        [pi.angular.coords.into(), pj.angular.coords.into()],
    );
    (
        Split3::new(outputs.se_lin_force.into(), outputs.se_ang_force.into()),
        Split3::new(outputs.se_lin_grad2.into(), outputs.se_ang_grad2.into()),
        Split3::from_angular(outputs.bt_ang_force.into()),
        Split3::from_angular(outputs.bt_ang_grad2.into()),
    )
}

#[tracked]
fn compute_host(
    radius: f32,
    young_modulus: f32,
    shear_modulus: f32,
    length: f32,
    rest_rotation: Vec4,
    [pi, pj]: [Vec3; 2],
    [qi, qj]: [Vec4; 2],
) -> Outputs {
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device(DeviceType::Cuda);

    let buffer = device.create_buffer::<Outputs>(1);

    let kernel = Kernel::<fn(f32, Vec4, Vec3, Vec3, Vec4, Vec4)>::new_with_options(
        &device,
        KernelBuildOptions {
            enable_fast_math: true,
            ..Default::default()
        },
        &track!(|length, rest_rotation, pi, pj, qi, qj| {
            let result = compute(
                radius,
                young_modulus,
                shear_modulus,
                length,
                rest_rotation,
                [pi, pj],
                [qi, qj],
            );
            buffer.write(0, result);
        }),
    );
    kernel.dispatch([1, 1, 1], &length, &rest_rotation, &pi, &pj, &qi, &qj);
    buffer.copy_to_vec()[0]
}

#[tracked]
fn conjugate(q: Expr<Vec4>) -> Expr<Vec4> {
    q * Vec3::splat_expr(-1.0).extend(1.0)
}

// Extracts the imaginary part of the p * q quaternion product
#[tracked]
fn qmul_imag(p: Expr<Vec4>, q: Expr<Vec4>) -> Expr<Vec3> {
    p.w * q.xyz() + q.w * p.xyz() + p.xyz().cross(q.xyz())
}

#[tracked]
fn qmul(p: Expr<Vec4>, q: Expr<Vec4>) -> Expr<Vec4> {
    qmul_imag(p, q).extend(p.w * q.w - p.xyz().dot(q.xyz()))
}

// Rotates a vector by a quaternion
#[tracked]
fn qapply(q: Expr<Vec4>, v: Expr<Vec3>) -> Expr<Vec3> {
    let t = 2.0 * q.xyz().cross(v);
    v + q.w * t + q.xyz().cross(t)
}

#[tracked]
fn qrotmat(q: Expr<Vec4>) -> Expr<Mat3> {
    let x2 = q.x + q.x;
    let y2 = q.y + q.y;
    let z2 = q.z + q.z;
    let xx = q.x * x2;
    let xy = q.x * y2;
    let xz = q.x * z2;
    let yy = q.y * y2;
    let yz = q.y * z2;
    let zz = q.z * z2;
    let wx = q.w * x2;
    let wy = q.w * y2;
    let wz = q.w * z2;

    Mat3::expr(
        Vec3::expr(1.0 - (yy + zz), xy + wz, xz - wy),
        Vec3::expr(xy - wz, 1.0 - (xx + zz), yz + wx),
        Vec3::expr(xz + wy, yz - wx, 1.0 - (xx + yy)),
    )
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value, PartialEq)]
struct Mat4x3 {
    a: Vec4,
    b: Vec4,
    c: Vec4,
}
impl Mat4x3 {
    fn expr(a: Expr<Vec4>, b: Expr<Vec4>, c: Expr<Vec4>) -> Expr<Self> {
        Self::from_comps_expr(Mat4x3Comps { a, b, c })
    }
}
impl From<Mat4x3> for Matrix3x4 {
    fn from(value: Mat4x3) -> Self {
        Matrix4x3::from_columns(&[value.a.into(), value.b.into(), value.c.into()]).transpose()
    }
}

#[tracked]
fn add_mat(s: Expr<Mat4x3>, t: Expr<Mat4x3>) -> Expr<Mat4x3> {
    Mat4x3::expr(s.a + t.a, s.b + t.b, s.c + t.c)
}

#[tracked]
fn sub_mat(s: Expr<Mat4x3>, t: Expr<Mat4x3>) -> Expr<Mat4x3> {
    Mat4x3::expr(s.a - t.a, s.b - t.b, s.c - t.c)
}

#[tracked]
fn mul_mat4x3(s: Expr<Mat4x3>, t: Expr<Mat4x3>) -> Expr<Mat3> {
    Mat3::expr(
        Vec3::expr(s.a.dot(t.a), s.b.dot(t.a), s.c.dot(t.a)),
        Vec3::expr(s.a.dot(t.b), s.b.dot(t.b), s.c.dot(t.b)),
        Vec3::expr(s.a.dot(t.c), s.b.dot(t.c), s.c.dot(t.c)),
    )
}

#[tracked]
fn mul_scalar(s: Expr<f32>, t: Expr<Mat4x3>) -> Expr<Mat4x3> {
    Mat4x3::expr(s * t.a, s * t.b, s * t.c)
}

#[tracked]
fn div_scalar(t: Expr<Mat4x3>, s: Expr<f32>) -> Expr<Mat4x3> {
    Mat4x3::expr(t.a / s, t.b / s, t.c / s)
}

#[tracked]
fn compute(
    radius: f32,
    young_modulus: f32,
    shear_modulus: f32,
    length: Expr<f32>,
    qrest: Expr<Vec4>,
    [pi, pj]: [Expr<Vec3>; 2],
    [qi, qj]: [Expr<Vec4>; 2],
) -> Expr<Outputs> {
    // Constants
    let bend_twist_coeff = {
        let i = PI * radius.powi(4) / 4.0;
        let j = PI * radius.powi(4) / 2.0;
        Vec3::new(young_modulus * i, young_modulus * i, shear_modulus * j)
    };

    let stretch_shear_coeff = {
        let s = PI * radius.powi(2);
        let a = 5.0_f32 / 6.0 * s;
        Vec3::new(shear_modulus * a, shear_modulus * a, young_modulus * s)
    };

    let g = {
        let q = qi / 2.0;
        Mat4x3::expr(
            Vec4::expr(q.w, -q.z, q.y, -q.x),
            Vec4::expr(q.z, q.w, -q.x, -q.y),
            Vec4::expr(-q.y, q.x, q.w, -q.z),
        )
    };

    let qm = (qi + qj) / 2.0;
    let qm_norm = qm.length();
    let qij = qm / qm_norm;
    let qijc = conjugate(qij);
    let pdiff = pj - pi;
    let qdiff = qj - qi;

    let darboux = 2.0 / length * qmul_imag(qijc, qdiff);

    let qijc_mat = {
        let q = qij;
        Mat4x3::expr(
            Vec4::expr(q.w, q.z, -q.y, -q.x),
            Vec4::expr(-q.z, q.w, q.x, -q.y),
            Vec4::expr(q.y, -q.x, q.w, -q.z),
        )
    };
    let a = {
        let q = qdiff;
        Mat4x3::expr(
            Vec4::expr(-q.w, -q.z, q.y, q.x),
            Vec4::expr(q.z, -q.w, -q.x, q.y),
            Vec4::expr(-q.y, q.x, -q.w, q.z),
        )
    };
    let b = {
        let m = Vec3::expr(
            qijc_mat.a.dot(qdiff),
            qijc_mat.b.dot(qdiff),
            qijc_mat.c.dot(qdiff),
        );
        Mat4x3::expr(m.x * qij, m.y * qij, m.z * qij)
    };

    // TODO: Can remove the 2.0 probably.
    let gradient_mul_length = sub_mat(
        div_scalar(sub_mat(a, b), qm_norm),
        mul_scalar(2.0_f32.expr(), qijc_mat),
    );

    // Actually gradient / length but that's shifted to later to reduce operations.
    let gradient = mul_mat4x3(gradient_mul_length, g);

    let bt_ang_force = -(gradient.transpose() * (bend_twist_coeff * darboux));

    let bt_ang_grad2 = Vec3::expr(
        gradient.x.dot(bend_twist_coeff * gradient.x),
        gradient.y.dot(bend_twist_coeff * gradient.y),
        gradient.z.dot(bend_twist_coeff * gradient.z),
    ) / (length * length);

    let qtotal = qmul(qij, qrest);

    // Inline these.
    let qtotal_mat = qrotmat(qtotal);
    let qtotal_mat_t = qtotal_mat.transpose();

    // Can also write the force using quaternion multiplication but that's less efficient.
    let strain = 1.0 / length * qtotal_mat_t * pdiff - Vec3::z();
    let se_lin_force = qtotal_mat * (stretch_shear_coeff * strain);
    let se_lin_grad2 = Vec3::expr(
        qtotal_mat_t.x.dot(stretch_shear_coeff * qtotal_mat_t.x),
        qtotal_mat_t.y.dot(stretch_shear_coeff * qtotal_mat_t.y),
        qtotal_mat_t.z.dot(stretch_shear_coeff * qtotal_mat_t.z),
    ) / (length * length);

    let qpart = qmul(pdiff.extend(0.0), qij);

    let qr = qrotmat(qij);
    let a = {
        let q = qpart;
        let m = [
            Vec3::expr(-q.w, q.z, -q.y),
            Vec3::expr(-q.z, -q.w, q.x),
            Vec3::expr(q.y, -q.x, -q.w),
            Vec3::expr(q.x, q.y, q.z),
        ];
        let qrt = qrotmat(qij).transpose();

        Mat4x3::expr(
            Vec4::expr(
                qrt.x.dot(m[0]),
                qrt.x.dot(m[1]),
                qrt.x.dot(m[2]),
                qrt.x.dot(m[3]),
            ),
            Vec4::expr(
                qrt.y.dot(m[0]),
                qrt.y.dot(m[1]),
                qrt.y.dot(m[2]),
                qrt.y.dot(m[3]),
            ),
            Vec4::expr(
                qrt.z.dot(m[0]),
                qrt.z.dot(m[1]),
                qrt.z.dot(m[2]),
                qrt.z.dot(m[3]),
            ),
        )
    };
    let b = Mat4x3::expr(pdiff.x * qij, pdiff.y * qij, pdiff.z * qij);

    // Actually the gradient / (length * qm_norm), but that's shifted to later to reduce operations.
    let gradient = qtotal_mat_t * mul_mat4x3(sub_mat(a, b), g);

    let se_ang_force = -(gradient.transpose() * (stretch_shear_coeff * strain)) / qm_norm;

    let se_ang_grad2 = Vec3::expr(
        gradient.x.dot(stretch_shear_coeff * gradient.x),
        gradient.y.dot(stretch_shear_coeff * gradient.y),
        gradient.z.dot(stretch_shear_coeff * gradient.z),
    ) / (length * length * qm_norm * qm_norm);

    Outputs::from_comps_expr(OutputsComps {
        se_lin_force,
        se_ang_force,
        se_lin_grad2,
        se_ang_grad2,
        bt_ang_force,
        bt_ang_grad2,
    })
}
