PYTHONPATH='.':$PYTHONPATH \
python tools/analyze_results.py \
    configs/gpsnet/panoptic_fpn_r50_fpn_1x_sgdet_psg.py \
    work_dirs/gpsnet_panoptic_fpn_r50_fpn_1x_sgdet_psg/result.pkl \  # specify annotation pickle file (from test)
    work_dirs/gpsnet_panoptic_fpn_r50_fpn_1x_sgdet_psg/analyze_viz \  # dir to save visualizations
    --topk 20 \
    --show-score-thr 0.3
