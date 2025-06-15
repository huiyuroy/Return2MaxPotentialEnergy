import gc
import os

import pyrdw.generator as generator
from common import data_path


def preprocess_scenes():
    """
    Preprocess all scenes. It's necessary to enable R2mpe algorithm.


    Returns:

    """
    p_path = data_path + "\\phy"
    p_files = [os.path.join(p_path, file) for file in os.listdir(p_path) if 'json' in file]

    for p_file in p_files:
        pscene = generator.load_scene_contour(p_file)
        pscene.update_segmentation()
        pscene.update_grids_precompute_attr(enable_vis=True,
                                            enable_vis_grid=True,
                                            enable_discrete=True)
        generator.save_scene(pscene, p_path)
        gc.collect()


if __name__ == '__main__':
    preprocess_scenes()

    print("--process done!--")
