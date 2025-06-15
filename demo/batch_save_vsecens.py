import gc
import os

import pyrdw.generator as generator
from common import data_path


def preprocess_scenes():
    """
    Preprocess all scenes. It's necessary to enable R2mpe algorithm.


    Returns:

    """
    v_path = data_path + "\\vir"
    v_files = [os.path.join(v_path, file) for file in os.listdir(v_path) if 'json' in file]

    for v_file in v_files:
        vscene = generator.load_scene_contour(v_file)
        vscene.update_segmentation()
        vscene.update_grids_precompute_attr(enable_vis=True,
                                            enable_vis_grid=False,
                                            enable_discrete=True)
        generator.save_scene(vscene, v_path)
        gc.collect()


if __name__ == '__main__':
    preprocess_scenes()

    print("--process done!--")
