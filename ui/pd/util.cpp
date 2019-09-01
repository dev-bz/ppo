#include "SpAlg.h"
#include "kin.hpp"
#include "util.hpp"
bool cRBDUtil::IsConstJointSubspace(const Eigen::MatrixXd &joint_mat, int j) {
	bool is_root = cKinTree::IsRoot(joint_mat, j);
	bool is_const = !is_root;
	return is_const;
}
Eigen::MatrixXd cRBDUtil::BuildJointSubspace(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &pose, int j) {
	cKinTree::eJointType j_type = cKinTree::GetJointType(joint_mat, j);
	bool is_root = cKinTree::IsRoot(joint_mat, j);
	Eigen::MatrixXd S;
	if (is_root) {
		S = BuildJointSubspaceRoot(joint_mat, pose);
	} else {
		switch (j_type) {
		case cKinTree::eJointTypeRevolute: S = BuildJointSubspaceRevolute(joint_mat, pose, j); break;
		case cKinTree::eJointTypePrismatic: S = BuildJointSubspacePrismatic(joint_mat, pose, j); break;
		case cKinTree::eJointTypePlanar: S = BuildJointSubspacePlanar(joint_mat, pose, j); break;
		case cKinTree::eJointTypeFixed: S = BuildJointSubspaceFixed(joint_mat, pose, j); break;
		case cKinTree::eJointTypeSpherical: S = BuildJointSubspaceSpherical(joint_mat, pose, j); break;
		default:
			assert(false); // unsupported joint type;
			break;
		}
	}
	return S;
}
void cRBDUtil::CalcWorldJointTransforms(const cRBDModel &model, Eigen::MatrixXd &out_trans_arr) {
	const Eigen::MatrixXd &joint_mat = model.mChar->GetJointMat();
	const Eigen::VectorXd &pose = model.mChar->GetPose();
	int num_joints = cKinTree::GetNumJoints(joint_mat);
	out_trans_arr.resize(num_joints * cSpAlg::gSVTransRows, cSpAlg::gSVTransCols);
	for (int j = 0; j < num_joints; ++j) {
		int row_idx = j * cSpAlg::gSVTransRows;
		int parent_id = cKinTree::GetParent(joint_mat, j);
		cSpAlg::tSpTrans parent_child_trans = model.GetSpParentChildTrans(j);
		cSpAlg::tSpTrans world_parent_trans = cSpAlg::BuildTrans();
		if (parent_id != cKinTree::gInvalidJointID) { world_parent_trans = cSpAlg::GetTrans(out_trans_arr, parent_id); }
		cSpAlg::tSpTrans world_child_trans = cSpAlg::CompTrans(parent_child_trans, world_parent_trans);
		out_trans_arr.block(row_idx, 0, cSpAlg::gSVTransRows, cSpAlg::gSVTransCols) = world_child_trans;
	}
}
Eigen::MatrixXd cRBDUtil::BuildJointSubspaceRoot(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &pose) {
	int dim = cKinTree::gRootDim;
	int pos_dim = cKinTree::gPosDim;
	int rot_dim = cKinTree::gRotDim;
	Eigen::MatrixXd S = Eigen::MatrixXd::Zero(cSpAlg::gSpVecSize, dim);
	tQuaternion quat = cKinTree::GetRootRot(joint_mat, pose);
	tMatrix E = cMathUtil::RotateMat(quat);
	S.block(3, 0, 3, pos_dim) = E.block(0, 0, 3, pos_dim).transpose();
	S.block(0, pos_dim, 3, rot_dim).setIdentity();
	return S;
}
Eigen::MatrixXd cRBDUtil::BuildJointSubspaceRevolute(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &pose,
																										 int j) {
	int dim = cKinTree::GetJointParamSize(cKinTree::eJointTypeRevolute);
	Eigen::MatrixXd S = Eigen::MatrixXd::Zero(cSpAlg::gSpVecSize, dim);
	S(2, 0) = 1;
	return S;
}
Eigen::MatrixXd cRBDUtil::BuildJointSubspacePrismatic(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &pose,
																											int j) {
	int dim = cKinTree::GetJointParamSize(cKinTree::eJointTypePrismatic);
	Eigen::MatrixXd S = Eigen::MatrixXd::Zero(cSpAlg::gSpVecSize, dim);
	S(3, 0) = 1;
	return S;
}
Eigen::MatrixXd cRBDUtil::BuildJointSubspacePlanar(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &pose,
																									 int j) {
	int dim = cKinTree::GetJointParamSize(cKinTree::eJointTypePlanar);
	Eigen::MatrixXd S = Eigen::MatrixXd::Zero(cSpAlg::gSpVecSize, dim);
	S(3, 0) = 1;
	S(4, 1) = 1;
	S(2, 2) = 1;
	return S;
}
Eigen::MatrixXd cRBDUtil::BuildJointSubspaceFixed(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &pose,
																									int j) {
	int dim = cKinTree::GetJointParamSize(cKinTree::eJointTypeFixed);
	Eigen::MatrixXd S = Eigen::MatrixXd::Zero(cSpAlg::gSpVecSize, dim);
	return S;
}
Eigen::MatrixXd cRBDUtil::BuildJointSubspaceSpherical(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &pose,
																											int j) {
	int dim = cKinTree::GetJointParamSize(cKinTree::eJointTypeSpherical);
	Eigen::MatrixXd S = Eigen::MatrixXd::Zero(cSpAlg::gSpVecSize, dim);
	S(0, 0) = 1;
	S(1, 1) = 1;
	S(2, 2) = 1;
	return S;
}
cSpAlg::tSpVec cRBDUtil::BuildCj(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &q,
																 const Eigen::VectorXd &q_dot, int j) {
	cKinTree::eJointType j_type = cKinTree::GetJointType(joint_mat, j);
	bool is_root = cKinTree::IsRoot(joint_mat, j);
	cSpAlg::tSpVec cj;
	if (is_root) {
		cj = BuildCjRoot(joint_mat, q, q_dot, j);
	} else {
		switch (j_type) {
		case cKinTree::eJointTypeRevolute: cj = BuildCjRevolute(joint_mat, q_dot, j); break;
		case cKinTree::eJointTypePrismatic: cj = BuildCjPrismatic(joint_mat, q_dot, j); break;
		case cKinTree::eJointTypePlanar: cj = BuildCjPlanar(joint_mat, q_dot, j); break;
		case cKinTree::eJointTypeFixed: cj = BuildCjFixed(joint_mat, q_dot, j); break;
		case cKinTree::eJointTypeSpherical: cj = BuildCjSpherical(joint_mat, q_dot, j); break;
		default:
			assert(false); // unsupported joint type;
			break;
		}
	}
	return cj;
}
cSpAlg::tSpVec cRBDUtil::BuildCjRoot(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &q,
																		 const Eigen::VectorXd &q_dot, int j) {
	tQuaternion quat = cKinTree::GetRootRot(joint_mat, q);
	tVector vel_lin = cKinTree::GetRootVel(joint_mat, q_dot);
	tVector vel_ang = cKinTree::GetRootAngVel(joint_mat, q_dot);
	Eigen::VectorXd joint_params;
	cKinTree::GetJointParams(joint_mat, q_dot, j, joint_params);
	tMatrix vel_dquat_mat = cMathUtil::BuildQuaternionDiffMat(quat);
	tVector dq_vec = vel_dquat_mat * vel_ang;
	tQuaternion dquat = cMathUtil::VecToQuat(dq_vec);
	tMatrix mat = tMatrix::Identity();
	mat(0, 0) = 4 * (quat.w() * dquat.w() + quat.x() * dquat.x());
	mat(1, 1) = 4 * (quat.w() * dquat.w() + quat.y() * dquat.y());
	mat(2, 2) = 4 * (quat.w() * dquat.w() + quat.z() * dquat.z());
	mat(1, 0) = 2 * (dquat.x() * quat.y() + quat.x() * dquat.y() - dquat.w() * quat.z() - quat.w() * dquat.z());
	mat(0, 1) = 2 * (dquat.x() * quat.y() + quat.x() * dquat.y() + dquat.w() * quat.z() + quat.w() * dquat.z());
	mat(2, 0) = 2 * (dquat.x() * quat.z() + quat.x() * dquat.z() + dquat.w() * quat.y() + quat.w() * dquat.y());
	mat(0, 2) = 2 * (dquat.x() * quat.z() + quat.x() * dquat.z() - dquat.w() * quat.y() - quat.w() * dquat.y());
	mat(2, 1) = 2 * (dquat.y() * quat.z() + quat.y() * dquat.z() - dquat.w() * quat.x() - quat.w() * dquat.x());
	mat(1, 2) = 2 * (dquat.y() * quat.z() + quat.y() * dquat.z() + dquat.w() * quat.x() + quat.w() * dquat.x());
	cSpAlg::tSpVec cj = cSpAlg::tSpVec::Zero();
	cj.segment(3, 3) = (mat * vel_lin).segment(0, 3);
	return cj;
}
cSpAlg::tSpVec cRBDUtil::BuildCjRevolute(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &q_dot, int j) {
	return cSpAlg::tSpVec::Zero();
}
cSpAlg::tSpVec cRBDUtil::BuildCjPrismatic(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &q_dot, int j) {
	return cSpAlg::tSpVec::Zero();
}
cSpAlg::tSpVec cRBDUtil::BuildCjPlanar(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &q_dot, int j) {
	return cSpAlg::tSpVec::Zero();
}
cSpAlg::tSpVec cRBDUtil::BuildCjFixed(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &q_dot, int j) {
	return cSpAlg::tSpVec::Zero();
}
cSpAlg::tSpVec cRBDUtil::BuildCjSpherical(const Eigen::MatrixXd &joint_mat, const Eigen::VectorXd &q_dot, int j) {
	return cSpAlg::tSpVec::Zero();
}
cSpAlg::tSpMat cRBDUtil::BuildMomentInertia(const Eigen::MatrixXd &body_defs, int part_id) {
	// inertia tensor of shape centered at the com
	assert(cKinTree::IsValidBody(body_defs, part_id));
	cKinTree::eBodyShape shape = cKinTree::GetBodyShape(body_defs, part_id);
	cSpAlg::tSpMat I;
	switch (shape) {
	case cKinTree::eBodyShapeBox: I = BuildMomentInertiaBox(body_defs, part_id); break;
	case cKinTree::eBodyShapeCapsule: I = BuildMomentInertiaCapsule(body_defs, part_id); break;
	default:
		assert(false); // unsupported shape
		break;
	}
	return I;
}
cSpAlg::tSpMat cRBDUtil::BuildMomentInertiaBox(const Eigen::MatrixXd &body_defs, int part_id) {
	const cKinTree::tBodyDef &def = body_defs.row(part_id);
	double mass = cKinTree::GetBodyMass(body_defs, part_id);
	double sx = def(cKinTree::eBodyParam0);
	double sy = def(cKinTree::eBodyParam1);
	double sz = def(cKinTree::eBodyParam2);
	double x = mass / 12.0 * (sy * sy + sz * sz);
	double y = mass / 12.0 * (sx * sx + sz * sz);
	double z = mass / 12.0 * (sx * sx + sy * sy);
	cSpAlg::tSpMat I = cSpAlg::tSpMat::Zero();
	I(0, 0) = x;
	I(1, 1) = y;
	I(2, 2) = z;
	I(3, 3) = mass;
	I(4, 4) = mass;
	I(5, 5) = mass;
	return I;
}
cSpAlg::tSpMat cRBDUtil::BuildMomentInertiaCapsule(const Eigen::MatrixXd &body_defs, int part_id) {
	const cKinTree::tBodyDef &def = body_defs.row(part_id);
	double mass = cKinTree::GetBodyMass(body_defs, part_id);
	double r = def(cKinTree::eBodyParam0);
	double h = def(cKinTree::eBodyParam1);
	double c_vol = M_PI * r * r * h;
	double hs_vol = M_PI * 2.0 / 3.0 * r * r * r;
	double density = mass / (c_vol + 2 * hs_vol);
	double cm = c_vol * density;
	double hsm = hs_vol * density;
	double x = cm * (0.25 * r * r + (1.0 / 12.0) * cm * h * h) + 2 * hsm * (0.4 * r * r + 0.375 * r * h + 0.25 * h * h);
	double y = (0.5 * cm + 0.8 * hsm) * r * r;
	double z = x;
	cSpAlg::tSpMat I = cSpAlg::tSpMat::Zero();
	I(0, 0) = x;
	I(1, 1) = y;
	I(2, 2) = z;
	I(3, 3) = mass;
	I(4, 4) = mass;
	I(5, 5) = mass;
	return I;
}
cSpAlg::tSpMat cRBDUtil::BuildInertiaSpatialMat(const Eigen::MatrixXd &body_defs, int part_id) {
	cSpAlg::tSpMat Ic = BuildMomentInertia(body_defs, part_id);
	tMatrix E = tMatrix::Identity();
	tVector r = -cKinTree::GetBodyAttachPt(body_defs, part_id);
	cSpAlg::tSpTrans X = cSpAlg::BuildTrans(E, r);
	cSpAlg::tSpMat Io = cSpAlg::BuildSpatialMatF(X) * Ic * cSpAlg::BuildSpatialMatM(cSpAlg::InvTrans(X));
	return Io;
}
void cRBDUtil::BuildBiasForce(const cRBDModel &model, Eigen::VectorXd &out_bias_force) {
	Eigen::VectorXd acc = Eigen::VectorXd::Zero(model.mChar->GetNumDof());
	SolveInvDyna(model, acc, out_bias_force);
}
void cRBDUtil::SolveInvDyna(const cRBDModel &model, const Eigen::VectorXd &acc, Eigen::VectorXd &out_tau) {
	const Eigen::MatrixXd &joint_mat = model.mChar->GetJointMat();
	const Eigen::MatrixXd &body_defs = model.mChar->GetBodyDefs();
	const tVector &gravity = model.GetGravity();
	const Eigen::VectorXd &pose = model.mChar->GetPose();
	const Eigen::VectorXd &vel = model.mChar->GetVel();
	assert(joint_mat.rows() == body_defs.rows());
	assert(pose.rows() == vel.rows());
	assert(pose.rows() == acc.rows());
	assert(cKinTree::GetNumDof(joint_mat) == pose.rows());
	cSpAlg::tSpVec vel0 = cSpAlg::tSpVec::Zero();
	cSpAlg::tSpVec acc0 = cSpAlg::BuildSV(tVector::Zero(), -gravity);
	int num_joints = cKinTree::GetNumJoints(joint_mat);
	Eigen::MatrixXd vels = Eigen::MatrixXd(num_joints, cSpAlg::gSpVecSize);
	Eigen::MatrixXd accs = Eigen::MatrixXd(num_joints, cSpAlg::gSpVecSize);
	Eigen::MatrixXd fs = Eigen::MatrixXd(num_joints, cSpAlg::gSpVecSize);
	for (int j = 0; j < num_joints; ++j) {
		if (cKinTree::IsValidBody(body_defs, j)) {
			cSpAlg::tSpTrans parent_child_trans = model.GetSpParentChildTrans(j);
			cSpAlg::tSpTrans world_child_trans = model.GetSpWorldJointTrans(j);
			const auto S = model.GetJointSubspace(j);
			Eigen::VectorXd q;
			Eigen::VectorXd dq;
			Eigen::VectorXd ddq;
			cKinTree::GetJointParams(joint_mat, pose, j, q);
			cKinTree::GetJointParams(joint_mat, vel, j, dq);
			cKinTree::GetJointParams(joint_mat, acc, j, ddq);
			cSpAlg::tSpVec cj = BuildCj(joint_mat, q, dq, j);
			cSpAlg::tSpVec vj = S * dq;
			cSpAlg::tSpMat I = BuildInertiaSpatialMat(body_defs, j);
			cSpAlg::tSpVec vel_p;
			cSpAlg::tSpVec acc_p;
			if (cKinTree::HasParent(joint_mat, j)) {
				int parent_id = cKinTree::GetParent(joint_mat, j);
				vel_p = vels.row(parent_id);
				acc_p = accs.row(parent_id);
			} else {
				vel_p = vel0;
				acc_p = acc0;
			}
			cSpAlg::tSpVec curr_vel = cSpAlg::ApplyTransM(parent_child_trans, vel_p) + vj;
			cSpAlg::tSpVec curr_acc =
					cSpAlg::ApplyTransM(parent_child_trans, acc_p) + S * ddq + cj + cSpAlg::CrossM(curr_vel, vj);
			cSpAlg::tSpVec curr_f = I * curr_acc + cSpAlg::CrossF(curr_vel, I * curr_vel);
			vels.row(j) = curr_vel;
			accs.row(j) = curr_acc;
			fs.row(j) = curr_f;
		}
	}
	out_tau = Eigen::VectorXd::Zero(pose.size());
	for (int j = num_joints - 1; j >= 0; --j) {
		if (cKinTree::IsValidBody(body_defs, j)) {
			cSpAlg::tSpVec curr_f = fs.row(j);
			const auto S = model.GetJointSubspace(j);
			Eigen::VectorXd curr_tau = S.transpose() * curr_f;
			cKinTree::SetJointParams(joint_mat, j, curr_tau, out_tau);
			if (cKinTree::HasParent(joint_mat, j)) {
				int parent_id = cKinTree::GetParent(joint_mat, j);
				cSpAlg::tSpTrans child_parent_trans = model.GetSpChildParentTrans(j);
				fs.row(parent_id) += cSpAlg::ApplyTransF(child_parent_trans, curr_f);
			}
		}
	}
}
void cRBDUtil::SolveForDyna(const cRBDModel &model, const Eigen::VectorXd &tau, Eigen::VectorXd &out_acc) {
	Eigen::VectorXd total_force = Eigen::VectorXd::Zero(model.mChar->GetNumDof());
	SolveForDyna(model, tau, total_force, out_acc);
}
void cRBDUtil::SolveForDyna(const cRBDModel &model, const Eigen::VectorXd &tau, const Eigen::VectorXd &total_force,
														Eigen::VectorXd &out_acc) {
	Eigen::VectorXd C;
	Eigen::MatrixXd H;
	BuildBiasForce(model, C);
	BuildMassMat(model, H);
	out_acc = H.ldlt().solve(tau + total_force - C);
}
void cRBDUtil::BuildMassMat(const cRBDModel &model, Eigen::MatrixXd &out_mass_mat) {
	const int svs = cSpAlg::gSpVecSize;
	int num_joints = model.mChar->GetNumJoints();
	Eigen::MatrixXd Is = Eigen::MatrixXd::Zero(num_joints * svs, svs);
	BuildMassMat(model, Is, out_mass_mat);
}
void cRBDUtil::BuildMassMat(const cRBDModel& model, Eigen::MatrixXd& inertia_buffer, Eigen::MatrixXd& out_mass_mat)
{
	// use composite-rigid-body algorithm
	const Eigen::MatrixXd& joint_mat = model.mChar->GetJointMat();
	const Eigen::MatrixXd& body_defs = model.mChar->GetBodyDefs();
	const Eigen::VectorXd& pose = model.mChar->GetPose();
	Eigen::MatrixXd& H = out_mass_mat;

	int dim = model.mChar->GetNumDof();
	int num_joints = model.mChar->GetNumJoints();
	H.setZero(dim, dim);
	const int svs = cSpAlg::gSpVecSize;

	Eigen::MatrixXd child_parent_mats_F = Eigen::MatrixXd(svs * num_joints, svs);
	Eigen::MatrixXd parent_child_mats_M = Eigen::MatrixXd(svs * num_joints, svs);
	Eigen::MatrixXd& Is = inertia_buffer;
	for (int j = 0; j < num_joints; ++j)
	{
		if (cKinTree::IsValidBody(body_defs, j))
		{
			Is.block(j * svs, 0, svs, svs) = BuildInertiaSpatialMat(body_defs, j);
		}

		cSpAlg::tSpTrans child_parent_trans = model.GetSpChildParentTrans(j);
		cSpAlg::tSpMat child_parent_mat = cSpAlg::BuildSpatialMatF(child_parent_trans);
		cSpAlg::tSpMat parent_child_mat = cSpAlg::BuildSpatialMatM(cSpAlg::InvTrans(child_parent_trans));
		child_parent_mats_F.block(j * svs, 0, svs, svs) = child_parent_mat;
		parent_child_mats_M.block(j * svs, 0, svs, svs) = parent_child_mat;
	}

	for (int j = num_joints - 1; j >= 0; --j)
	{
		if (cKinTree::IsValidBody(body_defs, j))
		{
			const auto curr_I = Is.block(j * svs, 0, svs, svs);
			int parent_id = cKinTree::GetParent(joint_mat, j);
			if (cKinTree::HasParent(joint_mat, j))
			{
				cSpAlg::tSpTrans child_parent_trans = model.GetSpChildParentTrans(j);
				cSpAlg::tSpMat child_parent_mat = child_parent_mats_F.block(j * svs, 0, svs, svs);
				cSpAlg::tSpMat parent_child_mat = parent_child_mats_M.block(j * svs, 0, svs, svs);
				Is.block(parent_id * svs, 0, svs, svs) += child_parent_mat * curr_I * parent_child_mat;
			}

			const auto S = model.GetJointSubspace(j);
			int offset = cKinTree::GetParamOffset(joint_mat, j);
			int dim = cKinTree::GetParamSize(joint_mat, j);
			Eigen::MatrixXd F = curr_I * S;
			H.block(offset, offset, dim, dim) = S.transpose() * F;

			int curr_id = j;
			while (cKinTree::HasParent(joint_mat, curr_id)) {
				cSpAlg::tSpMat child_parent_mat = child_parent_mats_F.block(curr_id * svs, 0, svs, svs);
				F = child_parent_mat * F;
				curr_id = cKinTree::GetParent(joint_mat, curr_id);
				int curr_offset = cKinTree::GetParamOffset(joint_mat, curr_id);
				int curr_dim = cKinTree::GetParamSize(joint_mat, curr_id);
				const auto curr_S = model.GetJointSubspace(curr_id);
				H.block(offset, curr_offset, dim, curr_dim) = F.transpose() * curr_S;
				H.block(curr_offset, offset, curr_dim, dim) = H.block(offset, curr_offset, dim, curr_dim).transpose();
			}
		}
	}
}