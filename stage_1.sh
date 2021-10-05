./cabinet/bin/python annotate_corners_v1.1.py $1
./cabinet/bin/python bbox_annotation_using_points.py $1
./cabinet/bin/python heatmap_v3.py $1
