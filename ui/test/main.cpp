#include <Box2D/Box2D.h>
#include <stdio.h>
struct BodyInterface {
	virtual int getNumBodies() = 0;
	virtual int getNumJoints() = 0;
	virtual b2Body *GetBody(int part) = 0;
	virtual b2Joint *GetJoint(int part) = 0;
	virtual float32 GetTarget(int part) = 0;
};
struct Test : BodyInterface {
	int getNumBodies() { return 4; };
	int getNumJoints() { return 3; }
	b2Body *GetBody(int part) { return nullptr; }
	b2Joint *GetJoint(int part) { return nullptr; }
	float32 GetTarget(int part) { return 0.0; }
};
void init_pd(BodyInterface *bx);
void update_pd(BodyInterface *bx);
void clear_pd(BodyInterface *bx);
int main() {
	Test test;
	init_pd(&test);
	update_pd(&test);
	clear_pd(&test);
	printf("finish");
	return 0;
}