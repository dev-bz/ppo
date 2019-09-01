#include "ch.hpp"
#include "kin.hpp"
const Eigen::VectorXd &cCharacter::GetPose() const { return mPose; }
void cCharacter::SetPose(const Eigen::VectorXd &pose) { mPose = pose; }
const Eigen::VectorXd &cCharacter::GetVel() const { return mVel; }
void cCharacter::SetVel(const Eigen::VectorXd &vel) { mVel = vel; }
const Eigen::MatrixXd &cCharacter::GetJointMat() const { return mJointMat; }
int cCharacter::GetNumJoints() const { return cKinTree::GetNumJoints(mJointMat); }
const Eigen::MatrixXd &cCharacter::GetBodyDefs() const { return mBodyDefs; }
int cCharacter::GetNumDof() const { return cKinTree::GetNumDof(mJointMat); }
int cCharacter::GetParamOffset(int joint_id) const { return cKinTree::GetParamOffset(mJointMat, joint_id); }
int cCharacter::GetParamSize(int joint_id) const { return cKinTree::GetParamSize(mJointMat, joint_id); }
bool cCharacter::IsEndEffector(int joint_id) const { return cKinTree::IsEndEffector(mJointMat, joint_id); }