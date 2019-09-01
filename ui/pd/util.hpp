#pragma once
#include "rbd.hpp"
#include <Eigen/Eigen>
struct cRBDUtil {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	static bool IsConstJointSubspace(const Eigen::MatrixXd &joint_mat, int j);
	static Eigen::MatrixXd BuildJointSubspace(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &pose, int j);
	static void CalcWorldJointTransforms(const cRBDModel &model, Eigen::MatrixXd &out_trans_arr);
	static void SolveInvDyna(const cRBDModel &model, const Eigen::VectorXd &acc, Eigen::VectorXd &out_tau);
	static void SolveForDyna(const cRBDModel &model, const Eigen::VectorXd &tau, Eigen::VectorXd &out_acc);
	static void SolveForDyna(const cRBDModel &model, const Eigen::VectorXd &tau, const Eigen::VectorXd &total_force,
													 Eigen::VectorXd &out_acc);
	static void BuildMassMat(const cRBDModel &model, Eigen::MatrixXd &out_mass_mat);
	static void BuildMassMat(const cRBDModel &model, Eigen::MatrixXd &inertia_buffer, Eigen::MatrixXd &out_mass_mat);
	static void BuildBiasForce(const cRBDModel &model, Eigen::VectorXd &out_bias_force);
	static Eigen::MatrixXd BuildJointSubspaceRoot(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &pose);
	static Eigen::MatrixXd BuildJointSubspaceRevolute(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &pose,
																										int j);
	static Eigen::MatrixXd BuildJointSubspacePrismatic(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &pose,
																										 int j);
	static Eigen::MatrixXd BuildJointSubspacePlanar(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &pose, int j);
	static Eigen::MatrixXd BuildJointSubspaceFixed(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &pose, int j);
	static Eigen::MatrixXd BuildJointSubspaceSpherical(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &pose,
																										 int j);
	static cSpAlg::tSpVec BuildCj(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &q,
																const Eigen::VectorXd &q_dot, int j);
	static cSpAlg::tSpVec BuildCjRoot(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &q,
																		const Eigen::VectorXd &q_dot, int j);
	static cSpAlg::tSpVec BuildCjRevolute(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &q_dot, int j);
	static cSpAlg::tSpVec BuildCjPrismatic(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &q_dot, int j);
	static cSpAlg::tSpVec BuildCjPlanar(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &q_dot, int j);
	static cSpAlg::tSpVec BuildCjFixed(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &q_dot, int j);
	static cSpAlg::tSpVec BuildCjSpherical(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &q_dot, int j);
	static cSpAlg::tSpMat BuildMomentInertia(const Eigen::MatrixXd &body_defs, int part_id);
	static cSpAlg::tSpMat BuildMomentInertiaBox(const Eigen::MatrixXd &body_defs, int part_id);
	static cSpAlg::tSpMat BuildMomentInertiaCapsule(const Eigen::MatrixXd &body_defs, int part_id);
	// builds the spatial inertial matrix in the coordinate frame of the parent joint
	static cSpAlg::tSpMat BuildInertiaSpatialMat(const Eigen::MatrixXd &body_defs, int part_id);
};