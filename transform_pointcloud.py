import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData


def translate_along_vector(points, vector, distance):
    unit_vector = vector / np.linalg.norm(vector)
    translation = unit_vector * distance

    print(f"\n{'='*70}")
    print("STEP 1: TRANSLATION")
    print(f"{'='*70}")
    print(f"Direction vector: {vector}")
    print(f"Unit vector: {unit_vector}")
    print(f"Translation distance: {distance} units")
    print(f"Translation vector: {translation}")

    points = points + translation

    return points, translation


def rotate_around_vector(points, rotation_center, axis_vector, angle_degrees, clockwise=True):
    angle_rad = np.radians(angle_degrees)
    if clockwise:
        angle_rad = -angle_rad

    k = axis_vector / np.linalg.norm(axis_vector)

    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)

    print(f"\n{'='*70}")
    print("STEP 2: ROTATION")
    print(f"{'='*70}")
    print(f"Rotation center: {rotation_center}")
    print(f"Rotation axis (normalized): {k}")
    print(f"Rotation angle: {angle_degrees}° ({'clockwise' if clockwise else 'counterclockwise'})")
    print(f"Rotation matrix:\n{R}")

    points = points - rotation_center
    points = (R @ points.T).T
    points = points + rotation_center

    return points, R


def scale_reconstruction(points, scale_factor):
    print(f"\n{'='*70}")
    print("STEP 3: SCALING")
    print(f"{'='*70}")
    print(f"Scale factor: {scale_factor}")
    print(f"Original: 1 unit = 20 cm = 0.2 m")
    print(f"After scaling: 1 unit = 1 m")
    print(f"Effect: All coordinates multiplied by {scale_factor}")

    center = np.mean(points, axis=0)

    points = points - center
    points = points * scale_factor
    points = points + center * scale_factor

    return points


def validate_transformations(original_points, transformed_points, translation,
                            rotation_matrix, scale_factor, rotation_center):
    print(f"\n{'='*70}")
    print("STEP 4: VALIDATION")
    print(f"{'='*70}")

    num_validation_points = 5
    indices = np.random.choice(len(original_points), num_validation_points, replace=False)
    sample_points = original_points[indices]

    print(f"\nValidating with {num_validation_points} randomly selected points")
    print("\nTransformation sequence:")
    print(f"  1. Translate by: {translation}")
    print(f"  2. Rotate around center: {rotation_center}")
    print(f"  3. Scale by factor: {scale_factor}")

    transformed_points_actual = transformed_points

    errors = []
    print(f"\n{'Point':<8} {'Original':<35} {'Expected':<35} {'Actual':<35} {'Error'}")
    print("-" * 120)

    for i, original_pt in enumerate(sample_points):
        pt = original_pt + translation

        pt_centered = pt - rotation_center
        pt_rotated = rotation_matrix @ pt_centered
        pt = pt_rotated + rotation_center

        transformed_center = rotation_center + translation
        pt_centered = pt - transformed_center
        pt_scaled = pt_centered * scale_factor
        pt_final = pt_scaled + transformed_center * scale_factor

        distances = np.linalg.norm(transformed_points_actual - pt_final, axis=1)
        closest_idx = np.argmin(distances)
        closest_pt = transformed_points_actual[closest_idx]
        error = np.linalg.norm(closest_pt - pt_final)

        errors.append(error)

        orig_str = f"[{original_pt[0]:7.3f}, {original_pt[1]:7.3f}, {original_pt[2]:7.3f}]"
        exp_str = f"[{pt_final[0]:7.3f}, {pt_final[1]:7.3f}, {pt_final[2]:7.3f}]"
        act_str = f"[{closest_pt[0]:7.3f}, {closest_pt[1]:7.3f}, {closest_pt[2]:7.3f}]"
        print(f"{i+1:<8} {orig_str:<35} {exp_str:<35} {act_str:<35} {error:.6f}")

    avg_error = np.mean(errors)
    max_error = np.max(errors)

    print("\n" + "="*70)
    print("VALIDATION RESULTS:")
    print(f"  Average error: {avg_error:.8f} units")
    print(f"  Maximum error: {max_error:.8f} units")

    threshold = 0.001
    if max_error < threshold:
        print(f"  Status: ✓ PASSED (all errors < {threshold})")
        print("  Transformations are mathematically correct!")
    else:
        print(f"  Status: ⚠ WARNING (some errors >= {threshold})")
        print("  Note: Small errors may be due to floating point precision")
    print("="*70)

    return avg_error, max_error


def load_ply(filepath):
    plydata = PlyData.read(filepath)
    vertex = plydata['vertex']
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']
    points = np.column_stack((x, y, z))

    colors = None
    if 'red' in vertex and 'green' in vertex and 'blue' in vertex:
        r = vertex['red'] / 255.0
        g = vertex['green'] / 255.0
        b = vertex['blue'] / 255.0
        colors = np.column_stack((r, g, b))

    return points, colors


def save_ply(filepath, points, colors=None):
    vertex = np.array(
        [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )

    if colors is not None:
        colors_uint8 = (colors * 255).astype(np.uint8)
        vertex = np.array(
            [(points[i, 0], points[i, 1], points[i, 2],
              colors_uint8[i, 0], colors_uint8[i, 1], colors_uint8[i, 2])
             for i in range(points.shape[0])],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )

    from plyfile import PlyElement
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(filepath)


def visualize_point_cloud(points, colors=None, title="Point Cloud", point_size=1, save_path=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    if colors is not None:
        ax.scatter(points[::10, 0], points[::10, 1], points[::10, 2],
                   c=colors[::10], s=point_size, alpha=0.5)
    else:
        ax.scatter(points[::10, 0], points[::10, 1], points[::10, 2],
                   s=point_size, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def main():
    pcd_path = "sparse/0/points.ply"

    print("="*70)
    print("POINT CLOUD TRANSFORMATION AND VALIDATION")
    print("="*70)
    print(f"\nLoading point cloud from: {pcd_path}")

    points, colors = load_ply(pcd_path)
    print(f"Loaded {len(points)} points")

    original_points = points.copy()

    camera_position = np.array([0.0, 0.0, 0.0])
    camera_vector = np.array([1.0, 0.5, 0.2])

    print(f"\nSelected camera:")
    print(f"  Position: {camera_position}")
    print(f"  Viewing direction: {camera_vector}")

    print("\nVisualizing original point cloud...")
    print("Close the window to continue with transformations")
    visualize_point_cloud(points, colors, "Original Point Cloud", save_path="original_pointcloud.png")

    points_transformed = points.copy()

    print("\n" + "="*70)
    print("APPLYING TRANSFORMATIONS")
    print("="*70)

    points_transformed, translation = translate_along_vector(
        points_transformed, camera_vector, distance=5.0
    )

    points_transformed, rotation_matrix = rotate_around_vector(
        points_transformed, camera_position, camera_vector,
        angle_degrees=60, clockwise=True
    )

    points_transformed = scale_reconstruction(points_transformed, scale_factor=0.2)

    validate_transformations(
        original_points, points_transformed, translation,
        rotation_matrix, 0.2, camera_position
    )

    print("\nVisualizing transformed point cloud...")
    print("Close the window to continue")
    visualize_point_cloud(points_transformed, None, "Transformed Point Cloud (scaled to metric)", save_path="transformed_pointcloud.png")

    print("\nVisualizing side-by-side comparison...")
    points_comparison = points_transformed.copy()
    offset_distance = (np.max(points[:, 0]) - np.min(points[:, 0])) * 1.5
    points_comparison[:, 0] += offset_distance

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')

    if colors is not None:
        ax.scatter(points[::10, 0], points[::10, 1], points[::10, 2],
                   c=colors[::10], s=1, alpha=0.5, label='Original')
    else:
        ax.scatter(points[::10, 0], points[::10, 1], points[::10, 2],
                   s=1, alpha=0.5, label='Original')

    ax.scatter(points_comparison[::10, 0], points_comparison[::10, 1], points_comparison[::10, 2],
               c='blue', s=1, alpha=0.5, label='Transformed')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Original vs Transformed')
    ax.legend()
    plt.savefig("comparison_pointcloud.png", dpi=150, bbox_inches='tight')
    print("Saved visualization to comparison_pointcloud.png")
    plt.show()

    output_path = "transformed_cloud.ply"
    save_ply(output_path, points_transformed, colors)
    print(f"\nTransformed point cloud saved to: {output_path}")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
