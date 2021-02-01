import rospy
from ackermann_msgs.msg import AckermannDrive
# from carla_msgs.msg import CarlaEgoVehicleControl
import numpy as np
from util.util import euler_to_quaternion, quaternion_to_euler

class VehicleController():

    def __init__(self, model_name='gem'):
        # Publisher to publish the control input to the vehicle model
        self.controlPub = rospy.Publisher("/carla/ego_vehicle/ackermann_cmd", AckermannDrive, queue_size = 1)
        self.model_name = model_name

    def stop(self):
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = 0
        newAckermannCmd.steering_angle = 0
        self.controlPub.publish(newAckermannCmd)


    def execute(self, currentPose, targetPose):
        """
            This function takes the current state of the vehicle and 
            the target state to compute low-level control input to the vehicle
            Inputs: 
                currentPose: ModelState, the current state of vehicle
                targetPose: The desired state of the vehicle
        """

        currentEuler = quaternion_to_euler(currentPose[1][0], currentPose[1][1], currentPose[1][2], currentPose[1][3])
        curr_x = currentPose[0][0]
        curr_y = currentPose[0][1]


        target_x = targetPose[0]
        target_y = targetPose[1]
        target_orientation = targetPose[2]
        target_v = targetPose[3]
        
        k_s = 0.1
        k_ds = 1
        k_n = 0.1
        k_theta = 1

        #compute errors
        xError = (target_x - curr_x) * np.cos(currentEuler[2]) + (target_y - curr_y) * np.sin(currentEuler[2])
        yError = -(target_x - curr_x) * np.sin(currentEuler[2]) + (target_y - curr_y) * np.cos(currentEuler[2])
        thetaError = target_orientation-currentEuler[2]
        curr_v = np.sqrt(currentPose[2][0]**2 + currentPose[2][1]**2)
        vError = target_v - curr_v
        
        delta = k_n*yError # + k_theta*thetaError
        # Checking if the vehicle need to stop
        if target_v > 0:
            v = xError*k_s + vError*k_ds
        else:
            v = xError*k_s - 0.05*k_ds            

        #Send computed control input to vehicle
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = v
        newAckermannCmd.steering_angle = delta
        self.controlPub.publish(newAckermannCmd)

