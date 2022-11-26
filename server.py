import grpc
import test_pb2
import test_pb2_grpc

from concurrent import futures
import time

import extract_feats.opensmile as of
import extract_feats.librosa as lf
import models
import utils
from moviepy.editor import *
from predict import *

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class MsgServicer(test_pb2_grpc.MsgServiceServicer):

    def GetMsg(self, request, context):
        print("Received name: %s" % request.name)
        # 识别出客户端发送的跌倒识别结果和情绪识别的文件地址
        strlist = request.name.split(',')
        humanFallCheck = strlist[0]
        video_path = strlist[1]
        # 情绪识别
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile('test.mp3')
        audio_path = 'test.mp3'

        config = utils.parse_opt()
        model = models.load(config)
        nums = predict(config, audio_path, model)
        mood = config.class_labels[int(nums[0])]
        predictResult = nums[1]

        helpInfo = '一切正常'
        if (humanFallCheck == '跌倒'):
            if (mood == 'fear' or mood == 'sad'):
                helpInfo = '120救助'
        return test_pb2.MsgResponse(msg=humanFallCheck + ',' + mood + ',' + helpInfo)
        # return test_pb2.MsgResponse(msg='老人跌倒检测情况为： %s!\n老人情绪识别情况为：%s\n建议对老人采取措施：%s' % (humanFallCheck, mood, helpInfo))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    test_pb2_grpc.add_MsgServiceServicer_to_server(MsgServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
