#!/bin/bash
python3 ./DirectVoxGO/hw4_1.py --render_test $1 --dump_images $2  --render_only --config DirectVoxGO/configs/nerf/hotdog.py --ft_path ./hw4_1_f.tar
# TODO - run your inference Python3 code