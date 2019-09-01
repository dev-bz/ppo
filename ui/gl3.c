#if 1
// BEGIN_INCLUDE(all)
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <android/sensor.h>
#include <android_native_app_glue.h>
#include <errno.h>
#include <jni.h>
#include <math.h>
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "native-activity", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "native-activity", __VA_ARGS__))
void glPrintf(const char *fmt, const char *err) {
  LOGW(fmt, err);
}
/**
 * Our saved state data.
 */
struct saved_state {
  float angle;
  int32_t x;
  int32_t y;
  int32_t z;
};
/**
 * Shared state for our app.
 */
struct engine {
  struct android_app *app;
  ASensorManager *sensorManager;
  const ASensor *accelerometerSensor;
  ASensorEventQueue *sensorEventQueue;
  int animating;
  EGLDisplay display;
  EGLSurface surface;
  EGLContext context;
  int32_t width;
  int32_t height;
  struct saved_state state;
};
/**
 * Initialize an EGL context for the current display.
 */
// GLUES
static void __gluMakeIdentityf(GLfloat m[16]) {
  m[0 + 4 * 0] = 1;
  m[0 + 4 * 1] = 0;
  m[0 + 4 * 2] = 0;
  m[0 + 4 * 3] = 0;
  m[1 + 4 * 0] = 0;
  m[1 + 4 * 1] = 1;
  m[1 + 4 * 2] = 0;
  m[1 + 4 * 3] = 0;
  m[2 + 4 * 0] = 0;
  m[2 + 4 * 1] = 0;
  m[2 + 4 * 2] = 1;
  m[2 + 4 * 3] = 0;
  m[3 + 4 * 0] = 0;
  m[3 + 4 * 1] = 0;
  m[3 + 4 * 2] = 0;
  m[3 + 4 * 3] = 1;
}
void MatrixMul2(float *out, float *p, float *s);
#define __glPi 3.14159265358979323846
void gluPerspectivef(GLfloat *m, GLfloat fovy, GLfloat aspect, GLfloat zNear, GLfloat zFar) {
  // GLfloat m[];
  GLfloat sine, cotangent, deltaZ;
  GLfloat radians = (GLfloat)(fovy / 2.0f * __glPi / 180.0f);
  deltaZ = zFar - zNear;
  sine = (GLfloat)sin(radians);
  if ((deltaZ == 0.0f) || (sine == 0.0f) || (aspect == 0.0f)) {
    return;
  }
  cotangent = (GLfloat)(cos(radians) / sine);
  __gluMakeIdentityf(m);
  m[0] = cotangent / aspect;
  m[5] = cotangent;
  m[10] = -(zFar + zNear) / deltaZ;
  m[11] = -1.0f;
  m[14] = -2.0f * zNear * zFar / deltaZ;
  m[15] = 0;
  // // glMultMatrixf(&m[0][0]);
}
void glInitMatrix(float *out) {
  out[0] = out[5] = out[10] = out[15] = 1;
  out[1] = out[2] = out[3] = out[4] = out[6] = out[7] = out[8] = out[9] = out[11] = out[12] = out[13] = out[14] = 0;
}
void rotYMatrixx(float *out, float r) {
  out[0] = cosf(r);
  out[2] = sinf(r);
  out[8] = -out[2];
  out[10] = out[0];
  out[5] = 1;
  out[1] = out[4] = out[6] = out[9] = 0;
}
void rotXMatrixx(float *out, float r) {
  out[9] = sinf(r);
  out[10] = cosf(r);
  out[5] = out[10];
  out[6] = -out[9];
  out[0] = 1;
  out[1] = out[2] = out[4] = out[8] = 0;
}
void MatrixMul2(float *out, float *p, float *s) {
  int i, j, k;
  float e;
  for (i = 0; i < 4; ++i) {
    for (j = 0; j < 4; ++j) {
      e = 0;
      for (k = 0; k < 4; ++k) {
        e += p[i * 4 + k] * s[k * 4 + j];
      }
      out[i * 4 + j] = e;
    }
  }
}
void dRfromQ(float *R, const float *q) {
  // dAASSERT (q && R);
  // q = (s,vx,vy,vz)
  float qq1 = 2 * q[1] * q[1];
  float qq2 = 2 * q[2] * q[2];
  float qq3 = 2 * q[3] * q[3];
  R[0] = 1 - qq2 - qq3;
  R[1] = 2 * (q[1] * q[2] - q[0] * q[3]);
  R[2] = 2 * (q[1] * q[3] + q[0] * q[2]);
  R[3] = 0.0f;
  R[4] = 2 * (q[1] * q[2] + q[0] * q[3]);
  R[5] = 1 - qq1 - qq3;
  R[6] = 2 * (q[2] * q[3] - q[0] * q[1]);
  R[7] = 0.0f;
  R[8] = 2 * (q[1] * q[3] - q[0] * q[2]);
  R[9] = 2 * (q[2] * q[3] + q[0] * q[1]);
  R[10] = 1 - qq1 - qq2;
  R[11] = 0.0f;
}
void glRotatef(float *R, float qw, float qx, float qy, float qz) {
  // dAASSERT (q && R);
  // q = (s,vx,vy,vz)
  float qq1 = 2 * qx * qx;
  float qq2 = 2 * qy * qy;
  float qq3 = 2 * qz * qz;
  R[0] = 1 - qq2 - qq3;
  R[1] = 2 * (qx * qy - qw * qz);
  R[2] = 2 * (qx * qz + qw * qy);
  R[3] = 0.0f;
  R[4] = 2 * (qx * qy + qw * qz);
  R[5] = 1 - qq1 - qq3;
  R[6] = 2 * (qy * qz - qw * qx);
  R[7] = 0.0f;
  R[8] = 2 * (qx * qz - qw * qy);
  R[9] = 2 * (qy * qz + qw * qx);
  R[10] = 1 - qq1 - qq2;
  R[11] = 0.0f;
}
float gyro[16];
#include "box2dtest.h"
static int engine_init_display(struct engine *engine) {
  // initialize OpenGL ES and EGL
  /*
   * Here specify the attributes of the desired configuration.
   * Below, we select an EGLConfig with at least 8 bits per color
   * component compatible with on-screen windows
   */
  const EGLint attribs[] = {EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_BLUE_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_RED_SIZE, 8, EGL_DEPTH_SIZE, 8, EGL_NONE};
  EGLint w, h, dummy, format;
  EGLint numConfigs;
  EGLConfig config;
  EGLSurface surface;
  EGLContext context;
  EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  eglInitialize(display, 0, 0);
  /* Here, the application chooses the configuration it desires. In this
   * sample, we have a very simplified selection process, where we pick
   * the first EGLConfig that matches our criteria */
  eglChooseConfig(display, attribs, &config, 1, &numConfigs);
  /* EGL_NATIVE_VISUAL_ID is an attribute of the EGLConfig that is
   * guaranteed to be accepted by ANativeWindow_setBuffersGeometry().
   * As soon as we picked a EGLConfig, we can safely reconfigure the
   * ANativeWindow buffers to match, using EGL_NATIVE_VISUAL_ID. */
  eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &format);
  ANativeWindow_setBuffersGeometry(engine->app->window, 0, 0, format);
  surface = eglCreateWindowSurface(display, config, engine->app->window, NULL);
  int attrib[4] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE, 0};
  context = eglCreateContext(display, config, EGL_NO_CONTEXT, attrib);
  if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE) {
    LOGW("Unable to eglMakeCurrent");
    return -1;
  }
  eglQuerySurface(display, surface, EGL_WIDTH, &w);
  eglQuerySurface(display, surface, EGL_HEIGHT, &h);
  engine->display = display;
  engine->context = context;
  engine->surface = surface;
  engine->width = w;
  engine->height = h;
  engine->state.angle = 0;
  // Initialize GL state.
  glClearColor(0.3f, 0.4f, 0.5f, 1.f);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glDepthFunc(GL_LEQUAL);
  glClearDepthf(1.0f);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glViewport(0, 0, w, h);
  void debugCreate(int w, int h);
  debugCreate(w, h);
  glEnable(GL_CULL_FACE);
  // glRotatef(90, 1, 0, 0);
  return 0;
}
static int show_ui = 1;
static void engine_draw_frame(struct engine *engine) {
  if (engine->display == NULL) {
    // No display.
    return;
  }
#ifdef BOX2D
  box2d_step();
#endif
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#ifdef BOX2D
  if (show_ui)
    box2d_ui(engine->width, engine->height, engine->state.x, engine->state.y, engine->state.z, 0);
  box2d_draw();
  void debugFlush();
  debugFlush();
#endif
  glFlush();
  eglSwapBuffers(engine->display, engine->surface);
}
/**
 * Tear down the EGL context currently associated with the display.
 */
static void engine_term_display(struct engine *engine) {
  void debugDestroy();
  debugDestroy();
  if (engine->display != EGL_NO_DISPLAY) {
    eglMakeCurrent(engine->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (engine->context != EGL_NO_CONTEXT) {
      eglDestroyContext(engine->display, engine->context);
    }
    if (engine->surface != EGL_NO_SURFACE) {
      eglDestroySurface(engine->display, engine->surface);
    }
    eglTerminate(engine->display);
  }
  engine->animating = 0;
  engine->display = EGL_NO_DISPLAY;
  engine->context = EGL_NO_CONTEXT;
  engine->surface = EGL_NO_SURFACE;
}
/**
 * Process the next input event.
 */
static int32_t engine_handle_input(struct android_app *app, AInputEvent *event) {
  struct engine *engine = (struct engine *)app->userData;
  if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION) {
    engine->animating = 1;
    engine->state.x = AMotionEvent_getX(event, 0); // - engine->width * 0.5f;
    engine->state.y = AMotionEvent_getY(event, 0); // - engine->height * 0.5f;
    int32_t action = AMotionEvent_getAction(event);
    if (action == AMOTION_EVENT_ACTION_DOWN)
      engine->state.z = 1;
    else if (action == AMOTION_EVENT_ACTION_UP)
      engine->state.z = 0;
    return 1;
  } else if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_KEY) {
    int keyState = AKeyEvent_getAction(event);
    int key = AKeyEvent_getKeyCode(event);
    if (AKEY_EVENT_ACTION_DOWN == keyState) {
      switch (key) {
      case AKEYCODE_MENU:
        show_ui = !show_ui;
        return 1;
      }
    } else if (AKEY_EVENT_ACTION_UP == keyState) {
    }
  }
  return 0;
}
/**
 * Process the next main command.
 */
static void engine_handle_cmd(struct android_app *app, int32_t cmd) {
  struct engine *engine = (struct engine *)app->userData;
  switch (cmd) {
  case APP_CMD_SAVE_STATE:
    // The system has asked us to save our current state.  Do so.
    engine->app->savedState = malloc(sizeof(struct saved_state));
    *((struct saved_state *)engine->app->savedState) = engine->state;
    engine->app->savedStateSize = sizeof(struct saved_state);
    break;
  case APP_CMD_INIT_WINDOW:
    // The window is being shown, get it ready.
    if (engine->app->window != NULL) {
      engine_init_display(engine);
      engine_draw_frame(engine);
    }
    break;
  case APP_CMD_TERM_WINDOW:
    // The window is being hidden or closed, clean it up.
    engine_term_display(engine);
    break;
  case APP_CMD_GAINED_FOCUS:
    // When our app gains focus, we start monitoring the accelerometer.
    if (engine->accelerometerSensor != NULL) {
      ASensorEventQueue_enableSensor(engine->sensorEventQueue, engine->accelerometerSensor);
      // We'd like to get 60 events per second (in us).
      ASensorEventQueue_setEventRate(engine->sensorEventQueue, engine->accelerometerSensor, (1000L / 60) * 1000);
    }
    break;
  case APP_CMD_LOST_FOCUS:
    // When our app loses focus, we stop monitoring the accelerometer.
    // This is to avoid consuming battery while not being used.
    if (engine->accelerometerSensor != NULL) {
      ASensorEventQueue_disableSensor(engine->sensorEventQueue, engine->accelerometerSensor);
    }
    // Also stop animating.
    engine->animating = 0;
    engine_draw_frame(engine);
    break;
  }
}
/**
 * This is the main entry point of a native application that is using
 * android_native_app_glue.  It runs in its own thread, with its own
 * event loop for receiving input events and doing other things.
 */
void android_main(struct android_app *state) {
  struct engine engine;
  // Make sure glue isn't stripped.
  app_dummy();
  memset(&engine, 0, sizeof(engine));
  state->userData = &engine;
  state->onAppCmd = engine_handle_cmd;
  state->onInputEvent = engine_handle_input;
  engine.app = state;
  // Prepare to monitor accelerometer
  if (0) {
    engine.sensorManager = ASensorManager_getInstance();
    engine.accelerometerSensor = ASensorManager_getDefaultSensor(engine.sensorManager, 15);
    engine.sensorEventQueue = ASensorManager_createEventQueue(engine.sensorManager, state->looper, LOOPER_ID_USER, NULL, NULL);
  }
  if (state->savedState != NULL) {
    // We are starting with a previous saved state; restore from it.
    engine.state = *(struct saved_state *)state->savedState;
  }
  ANativeActivity_setWindowFlags(state->activity, 0x80, 0);
#ifdef BOX2D
  LOGW("box2d_init at %d", __LINE__);
  box2d_init();
  engine.animating = 1;
#endif
  // loop waiting for stuff to do.
  while (1) {
    // Read all pending events.
    int ident;
    int events;
    struct android_poll_source *source;
    // If not animating, we will block forever waiting for events.
    // If animating, we loop until all events are read, then continue
    // to draw the next frame of animation.
    while ((ident = ALooper_pollAll(engine.animating ? 0 : -1, NULL, &events, (void **)&source)) >= 0) {
      // Process this event.
      if (source != NULL) {
        source->process(state, source);
      }
      // If a sensor has data, process it now.
      if (ident == LOOPER_ID_USER) {
        if (engine.accelerometerSensor != NULL) {
          ASensorEvent event;
          while (ASensorEventQueue_getEvents(engine.sensorEventQueue, &event, 1) > 0) {
#ifdef BOX2D
            box2d_gravity(event.acceleration.x, event.acceleration.y);
#endif
            // gyro = event.uncalibrated_gyro;
            for (int i = 0; i < 16; ++i)
              gyro[i] = event.data[i];
            /*
            gyro.x_bias = event.magnetic.x;
            gyro.y_bias = event.magnetic.y;
            gyro.z_bias = event.magnetic.z;*/
            // LOGI("accelerometer: x=%f y=%f z=%f",
            // event.acceleration.x,
            // event.acceleration.y, event.acceleration.z);
          }
        }
      }
      // Check if we are exiting.
      if (state->destroyRequested != 0) {
#ifdef BOX2D
        LOGW("box2d_quit at %d", __LINE__);
        box2d_quit();
#endif
        engine_term_display(&engine);
        return;
      }
    }
    if (engine.animating) {
      // Done with events; draw next animation frame.
      engine.state.angle += .01f;
      if (engine.state.angle > 1) {
        engine.state.angle = 0;
      }
      // Drawing is throttled to the screen update rate, so there
      // is no need to do timing here.
      engine_draw_frame(&engine);
    }
  }
#ifdef BOX2D
  LOGW("box2d_quit at %d", __LINE__);
  box2d_quit();
#endif
}
// END_INCLUDE(all)
#endif