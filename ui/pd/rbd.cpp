#include "SpAlg.h"
#include "kin.hpp"
#include "rbd.hpp"
#include "util.hpp"
void cRBDModel::Init(cCharacter *ch, const tVector &gravity) {
	mGravity = gravity;
	mChar = ch;
	int num_dofs = mChar->GetNumDof();
	int num_joints = mChar->GetNumJoints();
	const int svs = cSpAlg::gSpVecSize;
	tMatrix trans_mat;
	InitJointSubspaceArr();
	mChildParentMatArr = Eigen::MatrixXd::Zero(num_joints * trans_mat.rows(), trans_mat.cols());
	mSpWorldJointTransArr = Eigen::MatrixXd::Zero(num_joints * cSpAlg::gSVTransRows, cSpAlg::gSVTransCols);
	mMassMat = Eigen::MatrixXd::Zero(num_dofs, num_dofs);
	mBiasForce = Eigen::VectorXd::Zero(num_dofs);
	mInertiaBuffer = Eigen::MatrixXd::Zero(num_joints * svs, svs);
}
void cRBDModel::Update() {
#if defined(ENABLE_RBD_PROFILER)
	printf("RBD_Update_BEG\n");
#endif
	UpdateJointSubspaceArr();
	UpdateChildParentMatArr();
	UpdateSpWorldTrans();
	UpdateMassMat();
	UpdateBiasForce();
#if defined(ENABLE_RBD_PROFILER)
	printf("RBD_Update_End\n\n");
#endif
}
const Eigen::MatrixXd &cRBDModel::GetMassMat() const { return mMassMat; }
const Eigen::VectorXd &cRBDModel::GetBiasForce() const { return mBiasForce; }
void cRBDModel::InitJointSubspaceArr() {
	int num_dofs = mChar->GetNumDof();
	int num_joints = mChar->GetNumJoints();
	mJointSubspaceArr = Eigen::MatrixXd(cSpAlg::gSpVecSize, num_dofs);
	for (int j = 0; j < num_joints; ++j) {
		int offset = cKinTree::GetParamOffset(mChar->mJointMat, j);
		int dim = cKinTree::GetParamSize(mChar->mJointMat, j);
		int r = static_cast<int>(mJointSubspaceArr.rows());
		mJointSubspaceArr.block(0, offset, r, dim) = cRBDUtil::BuildJointSubspace(mChar->mJointMat, mChar->mPose, j);
	}
}
void cRBDModel::UpdateJointSubspaceArr() {
#if defined(ENABLE_RBD_PROFILER)
	TIMER_PRINT_BEG(Update_Joint_Subspace)
#endif
	int num_joints = mChar->GetNumJoints();
	for (int j = 0; j < num_joints; ++j) {
		bool const_subspace = cRBDUtil::IsConstJointSubspace(mChar->mJointMat, j);
		if (!const_subspace) {
			int offset = cKinTree::GetParamOffset(mChar->mJointMat, j);
			int dim = cKinTree::GetParamSize(mChar->mJointMat, j);
			int r = static_cast<int>(mJointSubspaceArr.rows());
			mJointSubspaceArr.block(0, offset, r, dim) = cRBDUtil::BuildJointSubspace(mChar->mJointMat, mChar->mPose, j);
		}
	}
#if defined(ENABLE_RBD_PROFILER)
	TIMER_PRINT_END(Update_Joint_Subspace)
#endif
}
void cRBDModel::UpdateChildParentMatArr() {
#if defined(ENABLE_RBD_PROFILER)
	TIMER_PRINT_BEG(Update_Child_Parent_mat)
#endif
	int num_joints = mChar->GetNumJoints();
	for (int j = 0; j < num_joints; ++j) {
		tMatrix child_parent_trans = cKinTree::ChildParentTrans(mChar->mJointMat, mChar->mPose, j);
		int r = static_cast<int>(child_parent_trans.rows());
		int c = static_cast<int>(child_parent_trans.cols());
		mChildParentMatArr.block(j * r, 0, r, c) = child_parent_trans;
	}
#if defined(ENABLE_RBD_PROFILER)
	TIMER_PRINT_END(Update_Child_Parent_mat)
#endif
}
void cRBDModel::UpdateSpWorldTrans() {
#if defined(ENABLE_RBD_PROFILER)
	TIMER_PRINT_BEG(Update_SP_World_Trans)
#endif
	cRBDUtil::CalcWorldJointTransforms(*this, mSpWorldJointTransArr);
#if defined(ENABLE_RBD_PROFILER)
	TIMER_PRINT_END(Update_SP_World_Trans)
#endif
}
void cRBDModel::UpdateMassMat() {
#if defined(ENABLE_RBD_PROFILER)
	TIMER_PRINT_BEG(Update_Mass_Mat)
#endif
	cRBDUtil::BuildMassMat(*this, mInertiaBuffer, mMassMat);
#if defined(ENABLE_RBD_PROFILER)
	TIMER_PRINT_END(Update_Mass_Mat)
#endif
}
void cRBDModel::UpdateBiasForce() {
#if defined(ENABLE_RBD_PROFILER)
	TIMER_PRINT_BEG(Update_Bias_Force)
#endif
	cRBDUtil::BuildBiasForce(*this, mBiasForce);
#if defined(ENABLE_RBD_PROFILER)
	TIMER_PRINT_END(Update_Bias_Force)
#endif
}
tMatrix cRBDModel::GetChildParentMat(int j) const {
	assert(j >= 0 && j < mChar->GetNumJoints());
	tMatrix trans;
	int r = static_cast<int>(trans.rows());
	int c = static_cast<int>(trans.cols());
	trans = mChildParentMatArr.block(j * r, 0, r, c);
	return trans;
}
tMatrix cRBDModel::GetParentChildMat(int j) const {
	tMatrix child_parent_trans = GetChildParentMat(j);
	tMatrix parent_child_trans = cMathUtil::InvRigidMat(child_parent_trans);
	return parent_child_trans;
}
cSpAlg::tSpTrans cRBDModel::GetSpChildParentTrans(int j) const {
	tMatrix mat = GetChildParentMat(j);
	return cSpAlg::MatToTrans(mat);
}
cSpAlg::tSpTrans cRBDModel::GetSpParentChildTrans(int j) const {
	tMatrix mat = GetParentChildMat(j);
	return cSpAlg::MatToTrans(mat);
}
tMatrix cRBDModel::GetWorldJointMat(int j) const {
	cSpAlg::tSpTrans trans = GetSpWorldJointTrans(j);
	return cSpAlg::TransToMat(trans);
}
tMatrix cRBDModel::GetJointWorldMat(int j) const {
	cSpAlg::tSpTrans trans = GetSpJointWorldTrans(j);
	return cSpAlg::TransToMat(trans);
}
cSpAlg::tSpTrans cRBDModel::GetSpWorldJointTrans(int j) const {
	assert(j >= 0 && j < mChar->GetNumJoints());
	cSpAlg::tSpTrans trans = cSpAlg::GetTrans(mSpWorldJointTransArr, j);
	return trans;
}
cSpAlg::tSpTrans cRBDModel::GetSpJointWorldTrans(int j) const {
	cSpAlg::tSpTrans world_joint_trans = GetSpWorldJointTrans(j);
	return cSpAlg::InvTrans(world_joint_trans);
}
const Eigen::Block<const Eigen::MatrixXd> cRBDModel::GetJointSubspace(int j) const {
	assert(j >= 0 && j < mChar->GetNumJoints());
	int offset = cKinTree::GetParamOffset(mChar->mJointMat, j);
	int dim = cKinTree::GetParamSize(mChar->mJointMat, j);
	int r = static_cast<int>(mJointSubspaceArr.rows());
	return mJointSubspaceArr.block(0, offset, r, dim);
}
const tVector &cRBDModel::GetGravity() const { return mGravity; }