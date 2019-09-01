#pragma once
#include "ch.hpp"
#include "rbd.hpp"
#include "util/JsonUtil.h"
class cPDController {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	enum eParam {
		eParamJointID,
		eParamKp,
		eParamKd,
		eParamTargetTheta0,
		eParamTargetTheta1,
		eParamTargetTheta2,
		eParamTargetTheta3,
		eParamTargetTheta4,
		eParamTargetTheta5,
		eParamTargetTheta6,
		eParamTargetVel0,
		eParamTargetVel1,
		eParamTargetVel2,
		eParamTargetVel3,
		eParamTargetVel4,
		eParamTargetVel5,
		eParamTargetVel6,
		eParamUseWorldCoord,
		eParamMax
	};
	typedef Eigen::Matrix<double, eParamMax, 1> tParams;
	static bool LoadParams(const std::string &file, Eigen::MatrixXd &out_buffer);
	static bool ParsePDParams(const Json::Value &root, tParams &out_params);
};
struct cImpPDController {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	virtual void Init(std::shared_ptr<cCharacter> character, const Eigen::MatrixXd &pd_params, const tVector &gravity);
	virtual void Init(std::shared_ptr<cCharacter> character, const std::shared_ptr<cRBDModel> &model,
										const Eigen::MatrixXd &pd_params, const tVector &gravity);
	virtual void InitGains();
	virtual void UpdateRBDModel();
	virtual void CalcControlForces(double time_step, Eigen::VectorXd &out_tau);
	virtual void BuildTargetPose(Eigen::VectorXd &out_pose) const;
	virtual void BuildTargetVel(Eigen::VectorXd &out_vel) const;
	// int GetNumJoints();
	std::shared_ptr<cCharacter> mChar;
	Eigen::VectorXd mKp;
	Eigen::VectorXd mKd;
	Eigen::MatrixXd mPDParams;
	tVector mGravity;
	bool mExternRBDModel;
	std::shared_ptr<cRBDModel> mRBDModel;
};