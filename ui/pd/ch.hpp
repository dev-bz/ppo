#pragma once
#include <Eigen/Eigen>
struct cCharacter {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	virtual const Eigen::VectorXd &GetPose() const;
	virtual void SetPose(const Eigen::VectorXd &pose);
	virtual const Eigen::VectorXd &GetVel() const;
	virtual void SetVel(const Eigen::VectorXd &vel);
	virtual const Eigen::MatrixXd &GetJointMat() const;
	virtual const Eigen::MatrixXd &GetBodyDefs() const;
	virtual int GetNumDof() const;
	virtual int GetNumJoints() const;
	virtual int GetParamOffset(int joint_id) const;
	virtual int GetParamSize(int joint_id) const;
	virtual bool IsEndEffector(int joint_id) const;
	Eigen::MatrixXd mJointMat;
	Eigen::MatrixXd mBodyDefs;
	Eigen::VectorXd mPose, mVel;
};