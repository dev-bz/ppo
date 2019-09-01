#pragma once
#include "SpAlg.h"
#include "ch.hpp"
#include "util/MathUtil.h"
struct cRBDModel {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	virtual void Init(cCharacter *ch, const tVector &gravity);
	virtual void Update();
	virtual tMatrix GetChildParentMat(int j) const;
	virtual tMatrix GetParentChildMat(int j) const;
	virtual cSpAlg::tSpTrans GetSpChildParentTrans(int j) const;
	virtual cSpAlg::tSpTrans GetSpParentChildTrans(int j) const;
	virtual tMatrix GetWorldJointMat(int j) const;
	virtual tMatrix GetJointWorldMat(int j) const;
	virtual cSpAlg::tSpTrans GetSpWorldJointTrans(int j) const;
	virtual cSpAlg::tSpTrans GetSpJointWorldTrans(int j) const;
	virtual const Eigen::Block<const Eigen::MatrixXd> GetJointSubspace(int j) const;
	cCharacter *mChar;
	tVector mGravity;
	Eigen::MatrixXd mJointSubspaceArr;
	Eigen::MatrixXd mChildParentMatArr;
	Eigen::MatrixXd mSpWorldJointTransArr;
	Eigen::MatrixXd mMassMat;
	Eigen::VectorXd mBiasForce;
	Eigen::MatrixXd mInertiaBuffer;
	virtual const tVector &GetGravity() const;
	virtual const Eigen::MatrixXd &GetMassMat() const;
	virtual const Eigen::VectorXd &GetBiasForce() const;
	virtual void InitJointSubspaceArr();
	virtual void UpdateJointSubspaceArr();
	virtual void UpdateChildParentMatArr();
	virtual void UpdateSpWorldTrans();
	virtual void UpdateMassMat();
	virtual void UpdateBiasForce();
};