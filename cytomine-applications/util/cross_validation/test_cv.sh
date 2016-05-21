#!/bin/bash

/home/mass/GRD/r.mormont/miniconda/bin/python /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/cross_validation/pyxit_cross_validator.py \
    --cytomine_host "beta.cytomine.be" \
    --cytomine_public_key "ad014190-2fba-45de-a09f-8665f803ee0b" \
    --cytomine_private_key "767512dd-e66f-4d3c-bb46-306fa413a5eb" \
    --cytomine_base_path "/api/" \
    --cytomine_working_path "/home/mass/GRD/r.mormont/nobackup/cv" \
    --cytomine_id_software 152714969 \
    --cytomine_id_project 186829908 \
    --cytomine_selected_users 179077547 \
    --cytomine_binary "True" \
    --cytomine_excluded_terms 9444456 \
        --cytomine_excluded_terms 22042230 \
        --cytomine_excluded_terms 28792193 \
        --cytomine_excluded_terms 30559888 \
        --cytomine_excluded_terms 15054705 \
        --cytomine_excluded_terms 15054765 \
    --cytomine_positive_terms 676446 \
        --cytomine_positive_terms 676390 \
        --cytomine_positive_terms 676210 \
        --cytomine_positive_terms 676434 \
        --cytomine_positive_terms 676176 \
        --cytomine_positive_terms 676407 \
        --cytomine_positive_terms 15109451 \
        --cytomine_positive_terms 15109483 \
        --cytomine_positive_terms 15109489 \
        --cytomine_positive_terms 15109495 \
    --cytomine_negative_terms 675999 \
        --cytomine_negative_terms 676026 \
        --cytomine_negative_terms 933004 \
        --cytomine_negative_terms 8844862 \
        --cytomine_negative_terms 8844845 \
    --cytomine_excluded_annotations 30675573 \
        --cytomine_excluded_annotations 18107252 \
        --cytomine_excluded_annotations 9321884 \
        --cytomine_excluded_annotations 7994253 \
        --cytomine_excluded_annotations 9313842 \
    --cytomine_excluded_images 186836213 \
        --cytomine_excluded_images 186841715 \
        --cytomine_excluded_images 186841154 \
        --cytomine_excluded_images 186840535 \
        --cytomine_excluded_images 186842882 \
        --cytomine_excluded_images 186843325 \
        --cytomine_excluded_images 186843839 \
        --cytomine_excluded_images 186844344 \
        --cytomine_excluded_images 186844820 \
        --cytomine_excluded_images 186845164 \
    --cytomine_test_images 186842002 \
        --cytomine_test_images 186842285 \
    --cytomine_verbose 0 \
    --cv_images_out 1 \
    --pyxit_save_to "/home/mass/GRD/r.mormont/nobackup/models/test/test_pickle.pkl" \
    --pyxit_interpolation 1 \
    --pyxit_n_subwindows 2 \
    --pyxit_colorspace 2 \
    --pyxit_n_jobs 1 \
    --pyxit_min_size 0.6 \
    --pyxit_max_size 1.0 \
    --pyxit_dir_ls "/home/mass/GRD/r.mormont/nobackup/cv/test" \
    --forest_n_estimators 2 \
    --forest_min_samples_split 1 \
    --forest_max_features 16 \
    --svm 0 \
    --svm_c 1.0
