from interbotix_xs_modules.arm import InterbotixManipulatorXS

puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                                         robot_name=f'puppet_left', init_node=True)

torque_on(puppet_bot_left) #开启夹爪前必须激活夹爪！！！
puppet_bot_left.gripper.open()
puppet_bot_left.gripper.close()