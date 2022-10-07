import grpc
import numpy as np
import sys
sys.path.insert(0, "./prediction")
import asp_pb2 as aspmsg
import asp_pb2_grpc as asprpc
import asp_client as aspclt

class ASPClient(object):
    def __init__(self):
        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        channel = grpc.insecure_channel('localhost:50051', options=options)
        self.stub = asprpc.ASPStub(channel)

    def set_threshold(self, threshold):
        self.stub.set_threshold(aspmsg.DValue(value=threshold))

    def predict(self, imgpath, grasps):
        try:
            g2bytes = np.ndarray.tobytes(grasps.astype(float))
            out = self.stub.action_success_prediction(aspmsg.ASPInput(imgpath=imgpath, grasps=g2bytes))
            return np.frombuffer(out.probs, dtype=np.float32)
        except grpc.RpcError as rpc_error:
            print(f"[!] Prediction failed with {rpc_error.code()}")
            return

    def infer(self,imgpath, grasps):
        try:
            g2bytes = np.ndarray.tobytes(grasps.astype(float))             
            out = self.stub.action_success_prediction(aspmsg.ASPInput(imgpath=imgpath, grasps=g2bytes))
            res = self.stub.action_grasp_inference(out)
            return res.action, res.graspidx
        except grpc.RpcError as rpc_error:
            print(f'[!] Inference failed with {rpc_error.code()}')
            return

if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    imgpath = "./image/depth0.png"

    h_params = {"finger_length": 12,
                "finger_width":  6, 
                "open_width":    25}

    g_params = {"rotation_step": 45, 
                "depth_step":    25,
                "hand_depth":    10}


    print("Hand params: {}".format(h_params))
    print("FGE params: {}".format(g_params))

    from bpbot.binpicking import detect_grasp, draw_grasp
    grasps = detect_grasp(n_grasp=5, img_path=imgpath, 
                            g_params=g_params,
                            h_params=h_params)

    img_grasp = draw_grasp(grasps, imgpath, h_params, top_idx=-1)
    
    aspc = aspclt.ASPClient()
    aspc.set_threshold(0.5)
    g = grasps[:,0:2]
    # g = np.array([[4, 5], [6, 7]])
    
    # res_p = aspc.predict(imgpath=imgpath, grasps=g)

    a, g_idx = aspc.infer(imgpath=imgpath, grasps=g)
    print(f"Action Idx: {a}, Grasp Idx: {g_idx}")

    img_grasp = draw_grasp(grasps, imgpath, h_params, top_idx=g_idx)
    import cv2
    cv2.imshow("", img_grasp)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    end = timeit.default_timer()
    print("Time cost: ", end-start)
