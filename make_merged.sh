#!/bin/bash

for SPLIT in val train
do

mkdir -p merged/${SPLIT}

ls known_classes/images/${SPLIT} \
    | xargs -I {} ln -s ../../known_classes/images/${SPLIT}/{} merged/${SPLIT}/kc_{}

ls known_unknown_classes/images/${SPLIT} \
    | xargs -I {} ln -s ../../known_unknown_classes/images/${SPLIT}/{} merged/${SPLIT}/kuc_{}

done
