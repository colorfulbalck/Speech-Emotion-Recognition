import os
import numpy as np
import extract_feats.opensmile as of
import extract_feats.librosa as lf
import models
import utils
from moviepy.editor import *


def predict(config, audio_path: str, model):
    """
    预测音频情感

    Args:
        config: 配置项
        audio_path (str): 要预测的音频路径
        model: 加载的模型
    """

    # utils.play_audio(audio_path)

    if config.feature_method == 'o':
        # 一个玄学 bug 的暂时性解决方案
        of.get_data(config, audio_path, train=False)
        test_feature = of.load_feature(config, train=False)
    elif config.feature_method == 'l':
        test_feature = lf.get_data(config, audio_path, train=False)

    result = model.predict(test_feature)
    result_prob = model.predict_proba(test_feature)
    print('Recogntion: ', config.class_labels[int(result)])
    print('Probability: ', result_prob)
    # utils.radar(result_prob, config.class_labels)
    nums = [result, result_prob]
    return nums
    # return config.class_labels[int(result)]


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    video = VideoFileClip('test.avi')
    audio = video.audio
    audio.write_audiofile('test.mp3')
    audio_path = 'test.mp3'

    config = utils.parse_opt()
    model = models.load(config)
    predict(config, audio_path, model)
