#ifndef BOX2D
#define BOX2D
#ifdef __cplusplus
extern "C" {
#endif
struct display{float sx,sy,px,py,angle,r,g,b;};
void box2d_init();
void box2d_step();
void box2d_draw();
void box2d_ui(int width, int height, int mx, int my, unsigned char mbut,
              int scroll);
void box2d_quit();
void box2d_gravity(float x, float y);
//int box2d_info(float*out);
//int box2d_line(float*out);
#ifdef __cplusplus
}
#endif
#endif