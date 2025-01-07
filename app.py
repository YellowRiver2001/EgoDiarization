# -*- coding: utf-8 -*-

from importlib import import_module
import os
from flask_my_single_test_demo import run_diarization
#import cv2
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify


app = Flask(__name__)

# 全局变量用于存储处理结果
NAME = ""
FILE_FLAG = False
CAMERA_FLAG = False
RESULT_TEXT = ""

@app.route('/')
def index():
    """Video streaming home page."""
    # 将处理后的结果传递给模板
    return render_template('index.html', result_text=RESULT_TEXT)

def video_gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video', methods=['POST'])
# def upload():
#     f = request.files['file']
#     basepath = os.path.dirname(__file__)  # 当前文件所在路径
#     upload_path = './static/uploads'
#     if not os.path.exists(upload_path):
#         os.mkdir(upload_path)
#     upload_file_path = os.path.join(basepath, upload_path, f.filename)
#     f.save(upload_file_path)
    
#     # 设置全局变量
#     global NAME, FILE_FLAG, CAMERA_FLAG, RESULT_TEXT
#     NAME = upload_file_path
#     FILE_FLAG = True
#     CAMERA_FLAG = False

#     # 假设 `process_video` 是你的处理函数，返回文字结果
#     RESULT_TEXT = process_video(upload_file_path)  # 处理视频并返回文字结果

#     return redirect(url_for('index'))

@app.route('/video', methods=['POST'])
def upload():
    f = request.files['file']
    basepath = os.path.dirname(__file__)  # 当前文件所在路径
    upload_path = './static/uploads'
    if not os.path.exists(upload_path):
        os.mkdir(upload_path)
    upload_file_path = os.path.join(basepath, upload_path, f.filename)
    f.save(upload_file_path)
    
    # 设置全局变量
    global NAME, FILE_FLAG, CAMERA_FLAG, RESULT_TEXT
    NAME = upload_file_path
    FILE_FLAG = True
    CAMERA_FLAG = False

    # 假设 `process_video` 是你的处理函数，返回文字结果
    RESULT_TEXT = process_video(upload_file_path)  # 处理视频并返回文字结果

    return jsonify(result_text=RESULT_TEXT)

# def process_video(video_path):
#     """
#     处理视频的逻辑，返回一个字符串作为结果。
#     这里可以替换为实际的处理代码。
#     """
#     # 模拟视频处理返回的文字结果
#     fn = NAME.split('/')[-1].split('.')[0]
#     id,segments = run_diarization(fn)
#     result = "视频处理完成!返回日志结果。" + NAME
#     return [id,segments]

# def process_video(video_path):
#     """
#     处理视频的逻辑，返回一个字符串作为结果。
#     这里可以替换为实际的处理代码。
#     """
#     # 模拟视频处理返回的文字结果
#     fn = NAME.split('/')[-1].split('.')[0]
#     ids, segments = run_diarization(fn)  # 获取说话人ID和时间段
    
#     result = "视频处理完成! 返回日志结果。"
    
#     # 这里返回的是说话人ID和对应的时间段列表
#     formatted_result = []
#     for idx, segment in zip(ids, segments):
#         formatted_result.append(f"说话人 {idx}: 开始时间: {segment[0]}ms, 结束时间: {segment[1]}ms")
    
#     return formatted_result

def process_video(video_path):
    """
    处理视频的逻辑，返回一个字符串作为结果。
    这里可以替换为实际的处理代码。
    """
    # 模拟视频处理返回的文字结果
    fn = NAME.split('/')[-1].split('.')[0]
    id, segments = run_diarization(fn)  # 假设 id 和 segments 是返回的值

    # 格式化日志为一个列表，方便前端显示
    speaker_logs = []
    for i,segment in enumerate(segments):
        speaker_logs.append({
            'id': id[i],  # 说话人ID
            'start_time': segment[0],  # 语音段的开始时间（毫秒）
            'end_time': segment[1]     # 语音段的结束时间（毫秒）
        })
    
    result = "视频处理完成! 文件" + NAME.split('/')[-1] + "的日志结果如下："
    return [result, speaker_logs]


# 测试
# def process_video(video_path):
#     speaker_logs = [
#         {
#             'id': 0,
#             'start_time': 0,
#             'end_time': 1234
#         },
#         {
#             'id': 1,
#             'start_time': 465,
#             'end_time': 97878
#         },
#     ]
#     result = "视频处理完成! 文件" + NAME.split('/')[-1] + "的日志结果如下："
#     return [result, speaker_logs]


@app.route('/camera', methods=['POST'])
def camera_get():
    global CAMERA_FLAG, FILE_FLAG
    CAMERA_FLAG = True
    FILE_FLAG = False
    return redirect('/')
    
if __name__ == '__main__':
    #app.run(host='10.64.68.14', port=8080)
    #app.run(host='115.157.201.223', port=8080)
   app.run(host='0.0.0.0', port=8080)