
Egocentric Speaker Diarization with Vision-Guided Clustering and Adaptive Speech Re-detection

> [Egocentric Speaker Diarization with Vision-Guided Clustering and Adaptive Speech Re-detection](https://ieeexplore.ieee.org/abstract/document/10889699/) <br>
> [He Huang](https://ieeexplore.ieee.org/author/677931166674759), [Haoyuan Yu](https://yu-haoyuan.github.io/), [Daibo Liu](https://sites.google.com/site/dbliuuestc/), [Haowen Chen](http://csee.hnu.edu.cn/people/chenhaowen), [Minjie Cai](https://cai-mj.github.io/) <br>
> Hunan University <br>
> ICASSP 2025
---

Our paper is accepted by **ICASSP-2025**

![image](https://github.com/yu-haoyuan/EgoDiarization/blob/main/fig.png)


## Environment：
```python
pip install -r requirements.txt
```

Testing
---
Create a folder named `demo` inside the `Light-ASD` directory.
Place your test video file(s) within the newly created `demo` folder.
Then you will find
```
├── pyavi
│   ├── audio.wav (Audio from input video)
│   ├── video.avi (Copy of the input video)
│   ├── video_only.avi (Output video without audio)
│   └── video_out.avi  (Output video with audio)
├── pycrop (The detected face videos and audios)
│   ├── 000000.avi
│   ├── 000000.wav
│   ├── 000001.avi
│   ├── 000001.wav
│   └── ...
├── pyframes (All the video frames in this video)
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...	
└── pywork
    ├── faces.pckl (face detection result)
    ├── scene.pckl (scene detection result)
    ├── scores.pckl (ASD result)
    └── tracks.pckl (face tracking result)
```
in the demo

```
python main.py
```

our data is prepared, if you want to get the data from Ego4D dataset, you can use Light-ASD first to get speaker number probability

Model download
---  
download camplus.ckpt in https://drive.google.com/file/d/1CMrfXCiJT2VRIM1qAKEM-FWd3Cno_23C/view?usp=drive_link ,put it in the current folder.  
download sfd_face.pth in https://drive.google.com/file/d/1hd6QgCeJkeBWJ8rkTnOYSh3sthw_j8rH/view?usp=drive_link ,put it in the ./model/faceDetector/s3fd/ 


Citation
---
If this work or code is helpful in your research, please cite:

> @INPROCEEDINGS{10889699,author={Huang, He and Yu, Haoyuan and Liu, Daibo and Chen, Haowen and Cai, Minjie},
booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, title={Egocentric Speaker Diarization with Vision-Guided Clustering and Adaptive Speech Re-detection}, year={2025},volume={},number={},pages={1-5},keywords={Visualization;Adaptation models;Codes;Pipelines;Signal processing algorithms;Clustering algorithms;Signal processing;Noise measurement;Speech processing;Videos;audio-visual diarization;egocentric video},doi={10.1109/ICASSP49660.2025.10889699}}



## Acknowledgments：  
Thanks for the support of the following source repositories for this research：  
1.The speaker detection part of the code is modified from [this repository](https://github.com/Junhua-Liao/Light-ASD).  
2.The spectral clustering part of the code is modified from [this repository](https://gitee.com/Wilder_ting/speaker_diarization).  
3.The global tracking code is modified from [this repository](https://github.com/EGO4D/audio-visual).  



