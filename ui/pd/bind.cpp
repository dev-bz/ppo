#include "kin.hpp"
#include "pd.hpp"
#include <Box2D/Box2D.h>
std::shared_ptr<cImpPDController> pds[2];

struct BodyInterface {
  virtual int getNumBodies() = 0;
  virtual int GetPD() = 0;
  virtual b2Body *GetBody(int part) = 0;
  virtual b2Joint *GetJoint(int part) = 0;
  virtual float32 GetTarget(int part) = 0;
};
void init_pd(BodyInterface *bx) {
  auto &pd = pds[bx->GetPD()];
  if (pd.get())
    return;
  // float32 pow_scale[9] = {1.0, 1.0, 1.0, 0.65, 0.65, 1.0, 1.0, 0.65, 0.65};
  // float32 mov_scale[9] = {1.0, 1.0, 1.0, 0.65, 0.65, 1.0, 1.0, 0.65, 0.65};
  std::shared_ptr<cCharacter> ch(new cCharacter());
  Eigen::MatrixXd &mJointMat = ch->mJointMat;
  Eigen::MatrixXd &mBodyDefs = ch->mBodyDefs;
  mJointMat.resize(bx->getNumBodies(), cKinTree::eJointDescMax);
  mBodyDefs.resize(bx->getNumBodies(), cKinTree::eBodyParamMax);
  Eigen::MatrixXd pds;
  pds.resize(bx->getNumBodies(), cPDController::eParamMax);
  for (int i = 0; i < bx->getNumBodies(); ++i) {
    {
      auto *b = bx->GetBody(i);
      auto ct = b->GetWorldCenter() - b->GetPosition();
      cKinTree::tBodyDef def = cKinTree::BuildBodyDef();

      float ret = sqrtf(b->GetInertia() * 12 / b->GetMass() - 1);

      def(cKinTree::eBodyParamAttachX) = ct.x;
      def(cKinTree::eBodyParamAttachY) = ct.y;
      def(cKinTree::eBodyParam0) = ret;
      def(cKinTree::eBodyParam1) = 1.0f;
      def(cKinTree::eBodyParam2) = 1.0f;
      def(cKinTree::eBodyParamMass) = b->GetMass();
      mBodyDefs.row(i) = def;
    }
    {
      int pid = -1;
      tVector point(0.0, 0.0, 0.0, 0.0);
      auto j = bx->GetJoint(i);
      if (j) {
        auto p = j->GetBodyA();
        for (int k = 0; k < i; ++k)
          if (bx->GetBody(k) == p) {
            pid = k;
            break;
          }
        auto ct = j->GetBodyB()->GetPosition() - p->GetPosition();
        point.x() = ct.x;
        point.y() = ct.y;
      }
      cKinTree::tJointDesc desc = cKinTree::BuildJointDesc(i == 0 ? cKinTree::eJointTypeNone : cKinTree::eJointTypeRevolute, pid, point);
      mJointMat.row(i) = desc;
    }
    {
      cPDController::tParams p;
      p(cPDController::eParamKp) = 420.0f; // pow_scale[i];
      p(cPDController::eParamKd) = 42.0f;  //* mov_scale[i] / pow_scale[i];
      pds.row(i) = p;
    }
  }
  cKinTree::PostProcessJointMat(mJointMat);
  Eigen::VectorXd &default_pose = ch->mPose;
  cKinTree::BuildDefaultPose(mJointMat, default_pose);
  Eigen::VectorXd &default_vel = ch->mVel;
  cKinTree::BuildDefaultVel(mJointMat, default_vel);
  pd.reset(new cImpPDController());
  // cPDController::LoadParams("pd.json", pds);
  pd->Init(ch, pds, tVector(0, -10, 0, 0));
}
void update_pd(BodyInterface *bx) {
  auto &pd = pds[bx->GetPD()];
  Eigen::MatrixXd &mBodyDefs = pd->mChar->mBodyDefs;
  Eigen::MatrixXd &mJointMat = pd->mChar->mJointMat;
  Eigen::VectorXd &out_pose = pd->mChar->mPose;
  Eigen::VectorXd &out_vel = pd->mChar->mVel;
  Eigen::MatrixXd &mPDParams = pd->mPDParams;
  for (int i = 0; i < bx->getNumBodies(); ++i) {
    if (i > 0)
      mPDParams.row(i)(cPDController::eParamTargetTheta0) = bx->GetTarget(i);
    auto *b = bx->GetBody(i);
    if (b) {
      int param_offset = pd->mChar->GetParamOffset(i);
      int param_size = pd->mChar->GetParamSize(i);
      Eigen::VectorXd curr_pose = Eigen::VectorXd::Zero(param_size);
      Eigen::VectorXd curr_vel = Eigen::VectorXd::Zero(param_size);
      if (cKinTree::IsRoot(mJointMat, i)) {
        const b2Vec2 &p = b->GetPosition();
        const b2Vec2 &v = b->GetLinearVelocity();
        tQuaternion q = cMathUtil::AxisAngleToQuaternion(tVector(0, 0, 1, 0), b->GetAngle());
        curr_pose(0) = p.x;
        curr_pose(1) = p.y;
        curr_pose(cKinTree::gPosDim) = q.w();
        curr_pose(cKinTree::gPosDim + 1) = q.x();
        curr_pose(cKinTree::gPosDim + 2) = q.y();
        curr_pose(cKinTree::gPosDim + 3) = q.z();
        curr_vel(0) = v.x;
        curr_vel(1) = v.y;
      } else {
        auto *j = (b2RevoluteJoint *)bx->GetJoint(i);
        if (j) {
          curr_pose(0) = j->GetJointAngle();
          curr_vel(0) = j->GetJointSpeed();
        }
      }
      out_pose.segment(param_offset, param_size) = curr_pose;
      out_vel.segment(param_offset, param_size) = curr_vel;
    }
  }
  pd->UpdateRBDModel();
  Eigen::VectorXd out_tau = Eigen::VectorXd::Zero(pd->mChar->GetNumDof());
  pd->CalcControlForces(0.033, out_tau);
  for (int i = 1; i < bx->getNumBodies(); ++i) {
    auto *j = (b2RevoluteJoint *)bx->GetJoint(i);
    if (j) {
      int param_offset = pd->mChar->GetParamOffset(i);
      int param_size = pd->mChar->GetParamSize(i);
      float32 t = out_tau(param_offset);
      j->SetMotorSpeed(t > 0 ? 100 : -100);
      j->SetMaxMotorTorque(b2Abs(t));
    }
  }
  /*int cnt = out_tau.size();
  printf("out_tau:%d\n", cnt);
  for (int i = 0; i < cnt; ++i) { printf("out_tau[%d]: %f\n", i, out_tau[i]); }*/
}
void clear_pd(BodyInterface *bx) {
  auto &pd = pds[bx->GetPD()];
  pd.reset();
}