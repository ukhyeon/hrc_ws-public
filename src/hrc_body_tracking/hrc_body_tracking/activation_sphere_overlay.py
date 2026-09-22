import cv2
import numpy as np


def project_world_points_to_image(
    points_world,
    camera_rotation,
    camera_position_world,
    camera_matrix,
    distortion,
    near_clip=0.05,
):
    """Project world-frame 3D points into the distorted RGB image."""
    points_world = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    camera_rotation = np.asarray(camera_rotation, dtype=np.float32).reshape(3, 3)
    camera_position_world = np.asarray(
        camera_position_world,
        dtype=np.float32,
    ).reshape(1, 3)

    # Matches human_dynamics_node:
    # p_world = camera_rotation.T @ p_camera + camera_position_world.
    points_camera = (
        camera_rotation @ (points_world - camera_position_world).T
    ).T
    valid_depth = points_camera[:, 2] > near_clip

    pixels = np.full((points_camera.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid_depth):
        projected, _ = cv2.projectPoints(
            points_camera[valid_depth],
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            camera_matrix,
            distortion,
        )
        pixels[valid_depth] = projected.reshape(-1, 2)

    return pixels, valid_depth


def project_camera_points_to_image(
    points_camera,
    camera_matrix,
    distortion,
    near_clip=0.05,
):
    points_camera = np.asarray(points_camera, dtype=np.float32).reshape(-1, 3)
    valid_depth = points_camera[:, 2] > near_clip
    pixels = np.full((points_camera.shape[0], 2), np.nan, dtype=np.float32)

    if np.any(valid_depth):
        projected, _ = cv2.projectPoints(
            points_camera[valid_depth],
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            camera_matrix,
            distortion,
        )
        pixels[valid_depth] = projected.reshape(-1, 2)

    return pixels, valid_depth


def _sphere_silhouette_camera(center_camera, radius, sample_count=180):
    center_camera = np.asarray(center_camera, dtype=np.float32).reshape(3)
    distance = float(np.linalg.norm(center_camera))
    if distance <= radius:
        return None

    view_axis = center_camera / distance
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(reference, view_axis))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    basis_u = np.cross(view_axis, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(view_axis, basis_u)

    radius_ratio_sq = (radius / distance) ** 2
    circle_center = center_camera * (1.0 - radius_ratio_sq)
    circle_radius = radius * np.sqrt(1.0 - radius_ratio_sq)
    angle = np.linspace(
        0.0,
        2.0 * np.pi,
        sample_count,
        endpoint=True,
        dtype=np.float32,
    )
    return (
        circle_center[None, :]
        + circle_radius
        * (
            np.cos(angle)[:, None] * basis_u[None, :]
            + np.sin(angle)[:, None] * basis_v[None, :]
        )
    )


def project_sphere_wireframe(
    center_world,
    radius,
    camera_rotation,
    camera_position_world,
    camera_matrix,
    distortion,
    longitude_count=8,
    latitude_count=5,
    samples_per_circle=120,
):
    """Return a journal-style projected sphere overlay description."""
    center = np.asarray(center_world, dtype=np.float32).reshape(1, 3)
    camera_rotation = np.asarray(camera_rotation, dtype=np.float32).reshape(3, 3)
    camera_position_world = np.asarray(
        camera_position_world,
        dtype=np.float32,
    ).reshape(1, 3)
    angle = np.linspace(
        0.0,
        2.0 * np.pi,
        samples_per_circle,
        endpoint=True,
        dtype=np.float32,
    )
    curves_world = []

    for longitude in np.linspace(
        0.0,
        np.pi,
        longitude_count,
        endpoint=False,
        dtype=np.float32,
    ):
        curve = np.stack(
            [
                np.cos(angle) * np.cos(longitude),
                np.cos(angle) * np.sin(longitude),
                np.sin(angle),
            ],
            axis=1,
        )
        curves_world.append(center + radius * curve)

    latitudes = np.linspace(
        -0.75 * np.pi / 2.0,
        0.75 * np.pi / 2.0,
        latitude_count,
        dtype=np.float32,
    )
    for latitude in latitudes:
        z = np.sin(latitude)
        radial = np.cos(latitude)
        curve = np.stack(
            [
                radial * np.cos(angle),
                radial * np.sin(angle),
                np.full_like(angle, z),
            ],
            axis=1,
        )
        curves_world.append(center + radius * curve)

    projected_curves = []
    for curve_world in curves_world:
        pixels, valid = project_world_points_to_image(
            curve_world,
            camera_rotation,
            camera_position_world,
            camera_matrix,
            distortion,
        )
        projected_curves.append((pixels, valid))

    center_camera = (
        camera_rotation @ (center - camera_position_world).T
    ).T[0]
    silhouette_camera = _sphere_silhouette_camera(center_camera, radius)
    silhouette_pixels = None
    if silhouette_camera is not None:
        silhouette_pixels, silhouette_valid = project_camera_points_to_image(
            silhouette_camera,
            camera_matrix,
            distortion,
        )
        if not np.all(silhouette_valid):
            silhouette_pixels = None

    center_pixels, center_valid = project_camera_points_to_image(
        center_camera[None, :],
        camera_matrix,
        distortion,
    )

    return {
        "grid_curves": projected_curves,
        "silhouette": silhouette_pixels,
        "center": center_pixels[0] if center_valid[0] else None,
    }


def draw_projected_sphere_overlay(
    image,
    projection,
    source_resolution,
    radius,
):
    if projection is None:
        return image

    scale = np.array(
        [
            image.shape[1] / float(source_resolution["w"]),
            image.shape[0] / float(source_resolution["h"]),
        ],
        dtype=np.float32,
    )
    image_margin = max(image.shape[:2])
    fill_color = (185, 118, 32)
    grid_color = (225, 180, 105)
    outline_color = (255, 225, 165)

    silhouette = projection.get("silhouette")
    if silhouette is not None and np.isfinite(silhouette).all():
        silhouette_scaled = np.rint(silhouette * scale).astype(np.int32)
        fill_layer = image.copy()
        cv2.fillPoly(
            fill_layer,
            [silhouette_scaled],
            fill_color,
            lineType=cv2.LINE_AA,
        )
        cv2.addWeighted(fill_layer, 0.13, image, 0.87, 0.0, dst=image)

    grid_layer = image.copy()
    for pixels, valid in projection.get("grid_curves", []):
        for index in range(len(pixels) - 1):
            if not (valid[index] and valid[index + 1]):
                continue
            segment = pixels[index:index + 2] * scale
            if not np.isfinite(segment).all():
                continue
            if np.any(np.abs(segment) > image_margin * 4):
                continue
            cv2.line(
                grid_layer,
                tuple(np.rint(segment[0]).astype(np.int32)),
                tuple(np.rint(segment[1]).astype(np.int32)),
                grid_color,
                1,
                cv2.LINE_AA,
            )
    cv2.addWeighted(grid_layer, 0.32, image, 0.68, 0.0, dst=image)

    if silhouette is not None and np.isfinite(silhouette).all():
        silhouette_scaled = np.rint(silhouette * scale).astype(np.int32)
        cv2.polylines(
            image,
            [silhouette_scaled],
            True,
            outline_color,
            3,
            cv2.LINE_AA,
        )
        cv2.polylines(
            image,
            [silhouette_scaled],
            True,
            fill_color,
            1,
            cv2.LINE_AA,
        )

    center = projection.get("center")
    if center is not None and np.isfinite(center).all():
        center_scaled = tuple(np.rint(center * scale).astype(np.int32))
        cv2.circle(image, center_scaled, 5, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(image, center_scaled, 3, outline_color, -1, cv2.LINE_AA)
        base_label = "robot base"
        base_font_scale = 0.42
        base_thickness = 1
        base_x = int(center_scaled[0] + 9)
        base_y = int(center_scaled[1] - 9)
        cv2.putText(
            image,
            base_label,
            (base_x, base_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            base_font_scale,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            base_label,
            (base_x, base_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            base_font_scale,
            outline_color,
            base_thickness,
            cv2.LINE_AA,
        )

    label = f"Collision-assessment activation region | R_act = {radius:.3f} m"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    x0 = 18
    y0 = image.shape[0] - 18
    cv2.rectangle(
        image,
        (x0 - 8, y0 - text_size[1] - 9),
        (x0 + text_size[0] + 8, y0 + 7),
        (22, 25, 28),
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        (x0, y0),
        font,
        font_scale,
        outline_color,
        thickness,
        cv2.LINE_AA,
    )
    return image
