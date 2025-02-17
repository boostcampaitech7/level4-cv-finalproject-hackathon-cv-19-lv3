import mediapipe as mp

class mp_model:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
    
    def get_result(self, image):
        return self.pose.process(image)
