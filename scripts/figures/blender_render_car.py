"""Blender helper for scripts/figures/fig_car_relighting.py.

Run by Blender, not by the normal Python interpreter.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-json", type=Path, required=True)
    parser.add_argument("--camera", default="Camera Perspective")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=562)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--world-strength", type=float, default=1.0)
    parser.add_argument("--env-rotation-deg", type=float, default=0.0)
    parser.add_argument("--keep-scene-lights", action="store_true")
    parser.add_argument("--device", choices=("CPU", "GPU"), default="CPU")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    args = parser.parse_args(argv)
    return args


def configure_cycles(scene, args):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    if args.device == "GPU":
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for compute_type in ("OPTIX", "CUDA"):
                try:
                    prefs.compute_device_type = compute_type
                    break
                except TypeError:
                    continue
            prefs.get_devices()
            for device in prefs.devices:
                device.use = True
            scene.cycles.device = "GPU"
        except Exception as exc:
            print(f"[warn] Could not enable GPU rendering: {exc}; using CPU")
            scene.cycles.device = "CPU"


def configure_camera(scene, camera_name):
    camera = bpy.data.objects.get(camera_name)
    if camera is None or camera.type != "CAMERA":
        available = [obj.name for obj in bpy.data.objects if obj.type == "CAMERA"]
        raise ValueError(f"Camera '{camera_name}' not found. Available: {available}")
    scene.camera = camera


def configure_lights(keep_scene_lights):
    if keep_scene_lights:
        return
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            obj.hide_render = True


def configure_world(env_path, strength, rotation_deg):
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()

    tex_coord = tree.nodes.new(type="ShaderNodeTexCoord")
    mapping = tree.nodes.new(type="ShaderNodeMapping")
    env = tree.nodes.new(type="ShaderNodeTexEnvironment")
    background = tree.nodes.new(type="ShaderNodeBackground")
    output = tree.nodes.new(type="ShaderNodeOutputWorld")

    image = bpy.data.images.load(str(env_path), check_existing=False)
    try:
        image.colorspace_settings.name = "Linear Rec.709"
    except Exception:
        pass
    env.image = image
    background.inputs["Strength"].default_value = strength
    mapping.inputs["Rotation"].default_value[2] = math.radians(rotation_deg)

    tree.links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    tree.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    tree.links.new(env.outputs["Color"], background.inputs["Color"])
    tree.links.new(background.outputs["Background"], output.inputs["Surface"])


def render_job(scene, job, args):
    env_path = Path(job["env_path"])
    output_path = Path(job["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[render] {job.get('label', env_path.name)}")
    print(f"         env={env_path}")
    print(f"         out={output_path}")
    configure_world(env_path, args.world_strength, args.env_rotation_deg)
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    scene = bpy.context.scene
    configure_cycles(scene, args)
    configure_camera(scene, args.camera)
    configure_lights(args.keep_scene_lights)

    jobs = json.loads(args.jobs_json.read_text())
    for job in jobs:
        render_job(scene, job, args)


if __name__ == "__main__":
    main()
