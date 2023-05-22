import json
from pathlib import Path
from typing import Dict

import numpy as np

"""
Defines a new matrix world and saves it as a text file.
"""


def save_world(out_path: Path, world: np.array) -> None:
    wall_row = ("wallBlock " * (len(world[0]) + 2)).rstrip() + "\n"
    output_string = wall_row

    for line in world:
        output_string += "wallBlock "

        for block in line:
            output_string += block + " "

        output_string += "wallBlock\n"

    output_string += wall_row

    with open(out_path, "w") as f:
        f.write(output_string)


def create_world(out_path: Path, definition: Dict, multiplication: int = None) -> None:
    world = np.full((definition["height"], definition["width"]), definition["block"])
    for part in definition["parts"]:
        anchor_x, anchor_y = part["anchor"]["x"], part["anchor"]["y"]
        height, width = part["height"], part["width"]

        world[anchor_x:anchor_x + height, anchor_y:anchor_y + width] = part["block"]

    if multiplication is not None:
        mult_axis = multiplication // 2
        h_stack = np.concatenate([world] * mult_axis, axis=1)
        world = np.concatenate([h_stack] * mult_axis)
    save_world(out_path, world)


if __name__ == '__main__':
    OUTPUT_PATH = Path(
        "/home/marz/Documents/Unief/thesis/code/unity/Morph/Morph/Assets/WorldFiles/iteration2_big8.json")

    DEFINITION_PATH = Path("/home/marz/Documents/Unief/thesis/code/unity/PythonUtils/world_creation/worlds/iteration2.json")
    MULTIPLICATION = 8
    ####################################################################################################################

    with open(DEFINITION_PATH, "r") as f:
        definition = json.load(f)

    create_world(OUTPUT_PATH, definition, MULTIPLICATION)
