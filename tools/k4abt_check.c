/*
 * k4abt_check: minimal Azure Kinect Body Tracking smoke test (SETUP.md 2.5).
 *
 * Steps: open device -> start depth camera -> create tracker -> process N frames.
 * Prints the number of bodies and the processing time (enqueue -> pop) per frame.
 * Uses only APIs present in both SDK 1.3 / BT 1.0 and SDK 1.4 / BT 1.1.
 *
 * Build:  gcc -O2 -Wall -o ~/hrc_ws/build/k4abt_check ~/hrc_ws/tools/k4abt_check.c -lk4a -lk4abt
 * Run:    ~/hrc_ws/build/k4abt_check [frames]      (default 30)
 *
 * Exit codes: 0 ok, 1 bad argument, 2 device, 3 calibration/cameras, 4 tracker create, 5 frame processing.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <k4a/k4a.h>
#include <k4abt.h>

#define TAG "[k4abt_check] "
#define CAPTURE_TIMEOUT_MS 2000
#define TRACKER_TIMEOUT_MS 5000

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

int main(int argc, char **argv)
{
    int n_frames = 30;
    if (argc > 1) {
        n_frames = atoi(argv[1]);
        if (n_frames <= 0) {
            fprintf(stderr, TAG "usage: %s [frames > 0]\n", argv[0]);
            return 1;
        }
    }

    int rc = 0;
    k4a_device_t device = NULL;
    k4abt_tracker_t tracker = NULL;
    int cameras_started = 0;

    uint32_t count = k4a_device_get_installed_count();
    printf(TAG "devices: %u\n", count);
    if (count == 0) {
        fprintf(stderr, TAG "FAIL(device): no Azure Kinect found\n");
        return 2;
    }
    if (k4a_device_open(0, &device) != K4A_RESULT_SUCCEEDED) {
        fprintf(stderr, TAG "FAIL(device): k4a_device_open failed (udev rules? another process using it?)\n");
        return 2;
    }
    char serial[64];
    size_t serial_size = sizeof(serial);
    if (k4a_device_get_serialnum(device, serial, &serial_size) != K4A_BUFFER_RESULT_SUCCEEDED)
        snprintf(serial, sizeof(serial), "?");
    printf(TAG "device opened, serial %s\n", serial);

    k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    config.color_resolution = K4A_COLOR_RESOLUTION_OFF;
    config.camera_fps = K4A_FRAMES_PER_SECOND_30;

    k4a_calibration_t calibration;
    if (k4a_device_get_calibration(device, config.depth_mode, config.color_resolution, &calibration)
        != K4A_RESULT_SUCCEEDED) {
        fprintf(stderr, TAG "FAIL(cameras): k4a_device_get_calibration failed\n");
        rc = 3;
        goto cleanup;
    }
    if (k4a_device_start_cameras(device, &config) != K4A_RESULT_SUCCEEDED) {
        fprintf(stderr, TAG "FAIL(cameras): k4a_device_start_cameras failed\n");
        rc = 3;
        goto cleanup;
    }
    cameras_started = 1;
    printf(TAG "cameras started (NFOV_UNBINNED, 30 fps, color off)\n");

    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    double t0 = now_ms();
    if (k4abt_tracker_create(&calibration, tracker_config, &tracker) != K4A_RESULT_SUCCEEDED) {
        fprintf(stderr, TAG "FAIL(tracker): k4abt_tracker_create failed "
                            "(GPU generation / CUDA / cuDNN / ONNX model -- see SETUP.md 2.5)\n");
        rc = 4;
        goto cleanup;
    }
    printf(TAG "tracker created in %.1f ms\n", now_ms() - t0);

    int with_body = 0;
    double sum_ms = 0.0;
    for (int i = 0; i < n_frames; i++) {
        k4a_capture_t capture = NULL;
        k4a_wait_result_t wr = k4a_device_get_capture(device, &capture, CAPTURE_TIMEOUT_MS);
        if (wr != K4A_WAIT_RESULT_SUCCEEDED) {
            fprintf(stderr, TAG "FAIL(frame %d): k4a_device_get_capture %s\n", i,
                    wr == K4A_WAIT_RESULT_TIMEOUT ? "timed out" : "failed");
            rc = 5;
            goto cleanup;
        }

        double t_start = now_ms();
        wr = k4abt_tracker_enqueue_capture(tracker, capture, TRACKER_TIMEOUT_MS);
        k4a_capture_release(capture);
        if (wr != K4A_WAIT_RESULT_SUCCEEDED) {
            fprintf(stderr, TAG "FAIL(frame %d): k4abt_tracker_enqueue_capture %s\n", i,
                    wr == K4A_WAIT_RESULT_TIMEOUT ? "timed out" : "failed");
            rc = 5;
            goto cleanup;
        }

        k4abt_frame_t body_frame = NULL;
        wr = k4abt_tracker_pop_result(tracker, &body_frame, TRACKER_TIMEOUT_MS);
        if (wr != K4A_WAIT_RESULT_SUCCEEDED) {
            fprintf(stderr, TAG "FAIL(frame %d): k4abt_tracker_pop_result %s\n", i,
                    wr == K4A_WAIT_RESULT_TIMEOUT ? "timed out" : "failed");
            rc = 5;
            goto cleanup;
        }
        double dt = now_ms() - t_start;
        uint32_t bodies = k4abt_frame_get_num_bodies(body_frame);
        k4abt_frame_release(body_frame);

        printf(TAG "frame %3d: bodies %u, %.1f ms%s\n", i, bodies, dt,
               i == 0 ? "   (first frame includes model warm-up)" : "");
        fflush(stdout);
        if (bodies > 0)
            with_body++;
        if (i > 0)
            sum_ms += dt;
    }

    if (n_frames > 1)
        printf(TAG "OK: %d frames, frames with body %d, mean %.1f ms (excluding frame 0)\n",
               n_frames, with_body, sum_ms / (n_frames - 1));
    else
        printf(TAG "OK: 1 frame, frames with body %d\n", with_body);

cleanup:
    if (tracker) {
        k4abt_tracker_shutdown(tracker);
        k4abt_tracker_destroy(tracker);
    }
    if (cameras_started)
        k4a_device_stop_cameras(device);
    if (device)
        k4a_device_close(device);
    return rc;
}
