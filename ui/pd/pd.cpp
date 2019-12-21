#include "kin.hpp"
#include "pd.hpp"
void cImpPDController::Init(std::shared_ptr<cCharacter> character, const Eigen::MatrixXd &pd_params,
														const tVector &gravity) {
	std::shared_ptr<cRBDModel> model = std::shared_ptr<cRBDModel>(new cRBDModel());
	model->Init(character.get(), gravity);
	Init(character, model, pd_params, gravity);
	mExternRBDModel = false;
}
void cImpPDController::Init(std::shared_ptr<cCharacter> character, const std::shared_ptr<cRBDModel> &model,
														const Eigen::MatrixXd &pd_params, const tVector &gravity) {
	mGravity = gravity;
	mRBDModel = model;
	mPDParams = pd_params;
	mChar = character;
	InitGains();
}
const std::string gPDControllersKey = "PDControllers";
const std::string gPDParamKeys[cPDController::eParamMax] = {
		"JointID",			"Kp",						"Kd",						"TargetTheta0", "TargetTheta1", "TargetTheta2",
		"TargetTheta3", "TargetTheta4", "TargetTheta5", "TargetTheta6", "TargetVel0",		"TargetVel1",
		"TargetVel2",		"TargetVel3",		"TargetVel4",		"TargetVel5",		"TargetVel6",		"UseWorldCoord"};
bool cPDController::LoadParams(const std::string &file, Eigen::MatrixXd &out_buffer) {
	std::ifstream f_stream(file);
	Json::CharReaderBuilder reader;
	Json::Value root;
	bool succ = Json::parseFromStream(reader, f_stream, &root, NULL);
	f_stream.close();
	if (succ && !root["Skeleton"].isNull()) {
		root = root["Skeleton"];
		if (!root[gPDControllersKey].isNull()) {
			const Json::Value &pd_controllers = root[gPDControllersKey];
			int num_ctrls = pd_controllers.size();
			out_buffer.resize(num_ctrls, eParamMax);
			for (int i = 0; i < num_ctrls; ++i) {
				tParams curr_params;
				const Json::Value &json_pd_ctrl = pd_controllers.get(i, 0);
				bool succ_def = ParsePDParams(json_pd_ctrl, curr_params);
				if (succ_def) {
					int joint_id = i;
					curr_params[eParamJointID] = i;
					out_buffer.row(i) = curr_params;
				} else {
					succ = false;
					break;
				}
			}
		}
	} else {
		printf("Failed to load PD controller parameters from %s\n", file.c_str());
	}
	return succ;
}
bool cPDController::ParsePDParams(const Json::Value &root, tParams &out_params) {
	bool succ = true;
	out_params.setZero();
	for (int i = 0; i < eParamMax; ++i) {
		const std::string &curr_key = gPDParamKeys[i];
		if (!root[curr_key].isNull() && root[curr_key].isNumeric()) {
			Json::Value json_val = root[curr_key];
			double val = json_val.asDouble();
			out_params[i] = val;
		}
	}
	return succ;
}
void cImpPDController::InitGains() {
	int num_dof = mChar->GetNumDof();
	mKp = Eigen::VectorXd::Zero(num_dof);
	mKd = Eigen::VectorXd::Zero(num_dof);
	for (int j = 1; j < mChar->GetNumJoints(); ++j) {
		// const cPDController &pd_ctrl = GetPDCtrl(j);
		if (/*pd_ctrl.IsValid()*/ 1) {
			int param_offset = mChar->GetParamOffset(j);
			int param_size = mChar->GetParamSize(j);
      cPDController::tParams p = mPDParams.row(j);
      double kp = p(cPDController::eParamKp);
      double kd = p(cPDController::eParamKd);
			mKp.segment(param_offset, param_size) = Eigen::VectorXd::Ones(param_size) * kp;
			mKd.segment(param_offset, param_size) = Eigen::VectorXd::Ones(param_size) * kd;
		}
	}
}
void cImpPDController::UpdateRBDModel() { mRBDModel->Update(); }
void cImpPDController::CalcControlForces(double time_step, Eigen::VectorXd &out_tau) {
	double t = time_step;
	const Eigen::VectorXd &pose = mChar->GetPose();
	const Eigen::VectorXd &vel = mChar->GetVel();
	Eigen::VectorXd tar_pose;
	Eigen::VectorXd tar_vel;
	BuildTargetPose(tar_pose);
	BuildTargetVel(tar_vel);
	Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kp_mat = mKp.asDiagonal();
	Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kd_mat = mKd.asDiagonal();
	/*for (int j = 0; j < GetNumJoints(); ++j) {
		const cPDController &pd_ctrl = GetPDCtrl(j);
		if (!pd_ctrl.IsValid() || !pd_ctrl.IsActive()) {
			int param_offset = mChar->GetParamOffset(j);
			int param_size = mChar->GetParamSize(j);
			Kp_mat.diagonal().segment(param_offset, param_size).setZero();
			Kd_mat.diagonal().segment(param_offset, param_size).setZero();
		}
	}*/
	Eigen::MatrixXd M = mRBDModel->GetMassMat();
	/*for (int i = 0; i < M.rows(); ++i) {
		Eigen::VectorXd S = M.row(i);
		for (int j = 0; j < S.size(); ++j) { printf(" %f", S(j)); }
		printf("\n");
	}*/
	const Eigen::VectorXd &C = mRBDModel->GetBiasForce();
	M.diagonal() += t * mKd;
	Eigen::VectorXd pose_inc;
	const Eigen::MatrixXd &joint_mat = mChar->GetJointMat();
	cKinTree::VelToPoseDiff(joint_mat, pose, vel, pose_inc);
	pose_inc = pose + t * pose_inc;
	cKinTree::PoseProcessPose(joint_mat, pose_inc);
	Eigen::VectorXd pose_err;
	cKinTree::CalcVel(joint_mat, pose_inc, tar_pose, 1, pose_err);
	Eigen::VectorXd vel_err = tar_vel - vel;
	Eigen::VectorXd acc = Kp_mat * pose_err + Kd_mat * vel_err - C;
#if defined(IMP_PD_CTRL_PROFILER)
	TIMER_RECORD_BEG(Solve)
#endif
	// int root_size = cKinTree::gRootDim;
	// int num_act_dofs = static_cast<int>(acc.size()) - root_size;
	// auto M_act = M.block(root_size, root_size, num_act_dofs, num_act_dofs);
	// auto acc_act = acc.segment(root_size, num_act_dofs);
	// acc_act = M_act.ldlt().solve(acc_act);
	acc = M.ldlt().solve(acc);
#if defined(IMP_PD_CTRL_PROFILER)
	TIMER_RECORD_END(Solve, mPerfSolveTime, mPerfSolveCount)
#endif
	out_tau += Kp_mat * pose_err + Kd_mat * (vel_err - t * acc);
}
void cImpPDController::BuildTargetPose(Eigen::VectorXd &out_pose) const {
	out_pose = Eigen::VectorXd::Zero(mChar->GetNumDof());
	// const auto& joint_mat = mChar->GetJointMat();
	// tVector root_pos = mChar->GetRootPos();
	// tQuaternion root_rot = mChar->GetRootRotation();
	// cKinTree::SetRootPos(joint_mat, root_pos, out_pose);
	// cKinTree::SetRootRot(joint_mat, root_rot, out_pose);
	for (int j = 1; j < mChar->GetNumJoints(); ++j) {
		// const cPDController &pd_ctrl = GetPDCtrl(j);
		if (/*pd_ctrl.IsValid()*/ 1) {
			// pd_ctrl.GetTargetTheta(curr_pose);
			int param_offset = mChar->GetParamOffset(j);
			int param_size = mChar->GetParamSize(j);
			Eigen::VectorXd curr_pose = mPDParams.row(j).segment(cPDController::eParamTargetTheta0, param_size);
			out_pose.segment(param_offset, param_size) = curr_pose;
		}
	}
}
void cImpPDController::BuildTargetVel(Eigen::VectorXd &out_vel) const {
	out_vel = Eigen::VectorXd::Zero(mChar->GetNumDof());
	// const auto& joint_mat = mChar->GetJointMat();
	// tVector root_vel = mChar->GetRootVel();
	// tVector root_ang_vel = mChar->GetRootAngVel();
	// cKinTree::SetRootVel(joint_mat, root_vel, out_vel);
	// cKinTree::SetRootAngVel(joint_mat, root_ang_vel, out_vel);
	for (int j = 1; j < mChar->GetNumJoints(); ++j) {
		// const cPDController &pd_ctrl = GetPDCtrl(j);
		if (/*pd_ctrl.IsValid()*/ 1) {
			// pd_ctrl.GetTargetVel(curr_vel);
			int param_offset = mChar->GetParamOffset(j);
			int param_size = mChar->GetParamSize(j);
			Eigen::VectorXd curr_vel = mPDParams.row(j).segment(cPDController::eParamTargetVel0, param_size);
			out_vel.segment(param_offset, param_size) = curr_vel;
		}
	}
}