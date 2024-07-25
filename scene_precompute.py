import os

import generator as generator


def preprocess_scenes():
    """
    Preprocess all scenes. It's necessary to enable R2mpe algorithm.


    Returns:

    """
    p_path = "E:\\polyspaces\\phy"
    v_path = "E:\\polyspaces\\vir"

    p_files = [[os.path.join(p_path, file) for file in os.listdir(p_path) if 'json' in file][0]]
    v_files = [[os.path.join(v_path, file) for file in os.listdir(v_path) if 'json' in file][0]]

    for p_file in p_files:
        pscene = generator.load_scene_contour(p_file)
        pscene.update_grids_base_attr()
        pscene.update_grids_visibility()
        pscene.update_grids_weights()
        pscene.update_grids_rot_occupancy(True)
        pscene.update_grids_runtime_attr()
        generator.save_scene(pscene,p_path)

    for v_file in v_files:
        vscene = generator.load_scene_contour(v_file)
        vscene.update_grids_base_attr()
        vscene.update_grids_visibility()
        vscene.update_grids_weights()
        vscene.update_grids_rot_occupancy(False)
        vscene.update_grids_runtime_attr()
        generator.save_scene(vscene, v_path)


if __name__ == '__main__':
    preprocess_scenes()
    print("--process done!--")

