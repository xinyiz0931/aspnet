import os
import grpc
import time
import numpy as np

import sys
sys.path.append("./")

import predictor.predictor_pb2 as pdmsg
import predictor.predictor_pb2_grpc as pdrpc

class PredictorClient(object):
    def __init__(self):
        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        channel = grpc.insecure_channel('localhost:50051', options=options)
        self.stub = pdrpc.PredictorStub(channel)

    def predict(self,imgpath, grasps):
        return self.stub.predict(pdmsg.Grasp(imgpath=imgpath,grasps=grasps))

if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    import predict_client as pdclt
    pdc = pdclt.PredictorClient()
    imgpath = "/home/xinyi/Workspace/myrobot/vision/depth/depth.png"

    # -------------------- random grasp points ----------------------
    y = np.array([[4, 5], [6, 7]])
    # ---------------------------------------------------------------

    # ----------- use Module I: Model-free Grasp Detection ----------
    from bpbot.binpicking import detect_grasp

    h_params = {"finger_height": 30,
                "finger_width":  12, 
                "open_width":    40}
    g_params = {"rotation_step": 22.5, 
                "depth_step":    50,
                "hand_depth":    50}
    grasps = detect_grasp(n_grasp=5, img_path=imgpath, 
                            g_params=g_params,
                            h_params=h_params)
    y = grasps
    # ---------------------------------------------------------------

    y2bytes=np.ndarray.tobytes(y)
    pdc.predict(imgpath=imgpath, grasps=y2bytes)

    end = timeit.default_timer()
    print("Time cost: ", end-start)
