import pyvista as pv
import numpy as np
import pickle
import isingModel


# Recording simulation frames to a list
def record_ising_animation(model, steps_per_frame=10, n_frames=100):
    """
    Record a list of color frames representing the 3D lattice.
    Each frame is a (N, 4) array of RGBA colors for each lattice node.
    """
    frames = []
    for frame in range(n_frames):
        for _ in range(steps_per_frame):
            model.metropolis_step()
        # Get lattice state and flatten it.
        lattice_cpu = model.lattice.get().flatten()
        # Build a color array: for instance, spin +1 -> opaque red; spin -1 -> transparent.
        colors = np.zeros((lattice_cpu.size, 4), dtype=np.uint8)
        colors[lattice_cpu == 1] = np.array([255, 0, 0, 255], dtype=np.uint8)
        colors[lattice_cpu == -1] = np.array([0, 0, 255, 255], dtype=np.uint8)
        frames.append(colors)
    return frames


def save_animation_frames(frames, filename="ising_animation.pkl"):
    """Save recorded frames to a file."""
    with open(filename, "wb") as f:
        pickle.dump(frames, f)


def load_animation_frames(filename="ising_animation.pkl"):
    """Load recorded frames from a file."""
    with open(filename, "rb") as f:
        frames = pickle.load(f)
    return frames


def interactive_viewer(frames, L):
    """
    Launch an interactive PyVista viewer with two sliders:
      - Frame slider: to select the recorded simulation frame.
      - Slices slider: to control how many z-slices (central slices) are rendered opaque.

    Points outside the selected slice region are made transparent.

    Parameters:
      frames: List of recorded color frames (each is an (N, 4) array) for a cubic L x L x L lattice.
      L: Lattice size.
    """
    # Create a full coordinate grid for the lattice.
    grid = np.indices((L, L, L)).transpose(1, 2, 3, 0).reshape(-1, 3).astype(np.float32)
    full_grid = grid.copy()

    # Store the full lattice color data from the frames.
    # (Assume each frame is an array of shape (N, 4) with RGBA values.)
    state = {"frame": 0, "slices": L}  # default: show all slices

    # Function to compute new colors based on the current frame and slice selection.
    def compute_colors():
        colors = frames[state["frame"]].copy()  # shape (N, 4)
        # Determine the desired z-slice range (central slices).
        mid = L // 2
        s = state["slices"]
        lower = mid - s // 2
        upper = mid + (s + 1) // 2 - 1
        # Create a mask for points with z in [lower, upper].
        mask = (full_grid[:, 2] >= lower) & (full_grid[:, 2] <= upper)
        # For points outside the slice range, set alpha to 0.
        colors[~mask, 3] = 0
        return colors

    # Create the initial color array.
    initial_colors = compute_colors()

    # Create a PyVista PolyData object for the full grid.
    cloud = pv.PolyData(full_grid)
    cloud["RGBA"] = initial_colors

    plotter = pv.Plotter()
    # plotter.enable_depth_peeling()
    plotter.add_mesh(cloud, scalars="RGBA", rgb=True,
                     render_points_as_spheres=True, point_size=5)

    # Slider callback for frame selection.
    def frame_slider_callback(val):
        state["frame"] = int(val)
        new_colors = compute_colors()
        cloud["RGBA"] = new_colors

    # Add frame slider (continuous update).
    plotter.add_slider_widget(
        frame_slider_callback,
        rng=[0, len(frames) - 1],
        value=0,
        title="Frame",
        pointa=(0.025, 0.1),
        pointb=(0.31, 0.1),
        interaction_event="always"
    )

    # Slider callback for slices selection.
    def slice_slider_callback(val):
        state["slices"] = int(val)
        new_colors = compute_colors()
        cloud["RGBA"] = new_colors

    # Add slices slider: let slices vary from 1 to L.
    plotter.add_slider_widget(
        slice_slider_callback,
        rng=[1, L],
        value=L,
        title="Slices",
        pointa=(0.025, 0.2),
        pointb=(0.31, 0.2),
        interaction_event="always"
    )

    plotter.show()


def run_animate_ising_3d():
    # Create a 3D Ising model instance. For example, L=32, T=4.5 (near criticality).
    # model = isingModel.GPUIsingModelOptimized(L=100, dim=3, T=2.0, J=1.0, seed=42)
    # frames = record_ising_animation(model, steps_per_frame=10, n_frames=100)
    # save_animation_frames(frames, filename="ising_animation.pkl")



    # Later, you can load them:
    # loaded_frames = load_animation_frames("ising_animation_2.pkl")
    loaded_frames = load_animation_frames("ising_animation_2.pkl")
    # Launch the interactive viewer with a time slider.
    # interactive_viewer(loaded_frames, L=64)
    interactive_viewer(loaded_frames, L=100)
