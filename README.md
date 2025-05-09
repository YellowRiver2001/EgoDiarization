
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

Model download
---  
download camplus.ckpt in `https://drive.google.com/file/d/1CMrfXCiJT2VRIM1qAKEM-FWd3Cno_23C/view?usp=drive_link` ,put it in the current folder.  
download sfd_face.pth in `https://drive.google.com/file/d/1hd6QgCeJkeBWJ8rkTnOYSh3sthw_j8rH/view?usp=drive_link` ,put it in the `./model/faceDetector/s3fd/` 

Demo
---
one video save in `Light-ASD/demo` and
```
python demo.py --videoName 0001.avi
```
can get a demo output

Evaluation
---
Create a folder named `demo` inside the `Light-ASD` directory.
Place your test video file(s) within the newly created `demo` folder.
```
python Light-ASD/Ego4d_global_demo_final.py
```
Then you will find
```
├── pyavi
│   ├──speaker_global_EGO4d.txt(is used in next step)
├── pycrop
├── pyframes
└── pywork
```
in the demo, and we already put all `speaker_global_EGO4d.txt` in the `dataset/asd`
and you need to put ego4d dataset's audio in the exp like this `exp/test1/mid/0d4efcc9-a336-46f9-a1db-0623b5be2869`
then 
```
python main.py
```
you can get the DER output





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



