import gymnasium as gym
import numpy as np
from pydrake.all import *
from pydrake.gym import DrakeGymEnv

from manipulation.scenarios import AddShape, SetTransparency
from manipulation.utils import ConfigureParser,FindDataResource, RenderDiagram, running_as_notebook


gym.envs.register(
    id="Franka-v1",
    entry_point=("Franka_gym_pos:make_Franka_env"),
)

def AddFranka(plant):

    parser = Parser(plant)
    ConfigureParser(parser)
    panda_model = parser.AddModelsFromUrl("package://drake_models/franka_description/urdf/panda_arm.urdf")[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0"))
    
    peg_model = parser.AddModels("../drake_pih_sim/model/ee_peg/ee_peg_asm.sdf")[0]
    plant.WeldFrames(plant.GetFrameByName("panda_link8"), plant.GetFrameByName("ee_peg_top_center"))
    plant.set_gravity_enabled(peg_model, False)

    hole_model = parser.AddModels("../drake_pih_sim/model/hole/30mm_hole.sdf")[0]
    W_X_HOLE = RigidTransform(p=np.array([0.51794014, 0.01935995, -0.012]))
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("hole_bottom_center"), W_X_HOLE)


    return panda_model, peg_model, hole_model

        
class RewardSystem(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort("body_state", 14)
        self.DeclareVectorInputPort("actions", 1)
        self.DeclareVectorOutputPort("reward", 1, self.CalcReward)

    def CalcReward(self, context, output):
        panda_state = self.get_input_port(0).Eval(context)
        actions = self.get_input_port(1).Eval(context)
        output[0] = 0 

class TransformSystem(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        
        self.DeclareVectorInputPort("actions", 1)
        self.DeclareAbstractInputPort("body_poses_input", AbstractValue.Make([RigidTransform()]*12)) 
        self.DeclareAbstractOutputPort("transform_output", self.AllocateOutput, self.CalcOutput)

    def AllocateOutput(self):
        return AbstractValue.Make(RigidTransform())

    def CalcOutput(self, context, output):
        action = self.get_input_port(0).Eval(context)
        transforms_input = self.get_input_port(1).Eval(context)
        X_AE= transforms_input[9] # 9 = index of panda_link8

        if action[0] == 0:
            new_pos = np.array([0.001,0.,0.]) # x translation
        elif action[0] == 1:
            new_pos = np.array([-0.001,0.,0.]) 
        elif action[0] == 2:
            new_pos = np.array([0.,0.001,0.]) # y
        elif action[0] == 3:
            new_pos = np.array([0.,-0.001,0.]) 
        elif action[0] == 4:
            new_pos = np.array([0.,0.,0.001])  # z
        elif action[0] == 5:
            new_pos = np.array([0.,0.,-0.001]) 
       
        franka_pose = RigidTransform(p= new_pos + X_AE.translation(), R= X_AE.rotation())
        #PoseEE = RigidTransform(R=RotationMatrix(R=np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])), p=np.array([0, 0, 0.0585])) 
        #franka_pose = franka_pose.multiply(PoseEE.inverse())
       
        output.set_value(franka_pose)

def make_franka_move(meshcat=None):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
    # TODO(russt): randomize parameters.
    panda_model,peg_model,hole_model = AddFranka(plant)
    plant.Finalize()
    plant.set_name("plant")

    if meshcat:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        #meshcat.Set2dRenderMode(xmin=-0.35, xmax=0.35, ymin=-0.1, ymax=0.3)   

    # DIFFERENTIAL INVERSE KINEMATICS
    franka_context = plant.CreateDefaultContext()
    franka_ik_param = DifferentialInverseKinematicsParameters(7, 7)
    franka_ik_param.set_joint_position_limits([np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]), np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])])
    franka_ik_param.set_joint_velocity_limits([np.array([-2.1750, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100]), np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])])
    franka_ik = DifferentialInverseKinematicsIntegrator(robot=plant, frame_A=plant.world_frame(), frame_E=plant.GetFrameByName("panda_link8"), time_step=0.001, parameters=franka_ik_param, robot_context=franka_context)
    franka_ik_solver = builder.AddSystem(franka_ik)
    franka_ik_solver.set_name("franka_ik_solver")
    ####################

    # JOINT PID CONTROL
    kp = [7000] * plant.num_positions()
    ki = [1] * plant.num_positions()
    kd = [10] * plant.num_positions()

    #
    #builder.ExportInput(franka_ik_solver.GetInputPort("X_AE_desired"))
    #builder.ExportInput(franka_ik_solver.GetInputPort("use_robot_state"))
    #builder.ExportOutput(franka_ik_solver.get_output_port())
    builder.Connect(plant.get_state_output_port(panda_model), franka_ik_solver.GetInputPort("robot_state"))

    #

    franka_controller = builder.AddSystem(InverseDynamicsController(plant, kp, ki, kd, False))
    franka_controller.set_name("franka_controller")

    actions = builder.AddSystem(PassThrough(1))
    #positions_to_state = builder.AddSystem(Multiplexer([plant.num_positions(), plant.num_positions()]))
    #zeros = builder.AddSystem(ConstantVectorSource([0] * plant.num_positions()))

    #builder.Connect(actions.get_output_port(), positions_to_state.get_input_port(0))
    
    #builder.Connect(zeros.get_output_port(), positions_to_state.get_input_port(1))
    #builder.Connect(positions_to_state.get_output_port(),franka_controller.get_input_port_desired_state())
    builder.Connect(franka_controller.get_output_port_control(), plant.get_actuation_input_port())
    builder.Connect(plant.get_state_output_port(panda_model), franka_controller.get_input_port_estimated_state())

    builder.ExportOutput(plant.get_contact_results_output_port())
    builder.ExportInput(actions.get_input_port(), "actions")
    builder.ExportOutput(plant.get_state_output_port(), "observations")

    reward = builder.AddSystem(RewardSystem())
    transform = builder.AddSystem(TransformSystem())
    transform.set_name("transform")

    builder.Connect(plant.get_state_output_port(panda_model), reward.get_input_port(0))
    builder.Connect(actions.get_output_port(), reward.get_input_port(1))
    builder.Connect(actions.get_output_port(), transform.get_input_port(0))

    builder.Connect(transform.get_output_port(),franka_ik_solver.get_input_port(0)) #

    positions_to_state = builder.AddSystem(Multiplexer([plant.num_positions(), plant.num_positions()]))
    zeros = builder.AddSystem(ConstantVectorSource([0] * plant.num_positions()))

    builder.Connect(franka_ik_solver.get_output_port(), positions_to_state.get_input_port(0))
    builder.Connect(zeros.get_output_port(), positions_to_state.get_input_port(1))
    builder.Connect(positions_to_state.get_output_port(),franka_controller.get_input_port_desired_state())

    builder.Connect(plant.get_body_poses_output_port(), transform.get_input_port(1))
    builder.ExportOutput(reward.get_output_port(), "reward")

    diagram = builder.Build()
    diagram.set_name("Franka with PID Controller")
    simulator = Simulator(diagram)

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    controller_context = franka_controller.GetMyMutableContextFromRoot(context)
    franka_ik_solver_context = franka_ik_solver.GetMyMutableContextFromRoot(context)

    q0 = np.array([-1.57, 0.1, 0, -1.2, 0, 1.6, 0])
    #plant.SetPositions(plant_context, q0)
    plant.SetDefaultPositions(q0)

    diagram.ForcedPublish(context)

    x0 = np.hstack((q0, 0*q0)) # Initial State Variable
    plant.SetPositions(plant_context, q0)
    
    p0=np.array([0.51610254 ,0.00644409, 0.15879793])
    r0= Quaternion(np.array([7.06856869e-01, -3.00474510e-04 , 7.07353135e-01, -2.19491010e-03]))
    r0=RotationMatrix(r0)
    franka_init_pose = RigidTransform(p=p0, R=r0)
    
    #franka_ik_solver.GetInputPort("use_robot_state").FixValue(franka_ik_solver_context,True)
    #franka_controller.GetInputPort("desired_state").FixValue(controller_context, x0)

    #franka_joint_desired = franka_ik_solver.GetOutputPort("joint_positions").Eval(franka_ik_solver_context)
    #franka_joint_desired = np.hstack([franka_joint_desired, np.zeros(7)])
    #franka_controller.GetInputPort("desired_state").FixValue(controller_context, franka_joint_desired)

    
    return simulator

def make_Franka_env(meshcat=None):

    simulator = make_franka_move(meshcat=meshcat)
    #action_space = gym.spaces.Discrete(3)
    action_space = gym.spaces.Box(low=0, high= 5, shape=(1,), dtype=np.int32) 
    
    #clipped_action_space = np.clip(action_space, action_space.low, action_space.high)
    
    

    plant = simulator.get_system().GetSubsystemByName("plant")

    low = np.concatenate(
            (
                plant.GetPositionLowerLimits() ,
                plant.GetVelocityLowerLimits() ,
            )
        )
    high = np.concatenate(
            (
                plant.GetPositionUpperLimits() ,
                plant.GetVelocityUpperLimits() ,
            )
        )
    observation_space = gym.spaces.Box(
            low=np.asarray(low), high=np.asarray(high), dtype=np.float64
        )

    def custom_reset_handler(simulator, context, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        context.SetTime(0)
        simulator.Initialize()
        
        plant_context = plant.GetMyMutableContextFromRoot(context)
        initial_positions = np.array([-1.57, 0.1, 0, -1.2, 0, 2.6, 0])
        initial_velocities = np.zeros_like(initial_positions)
        
        plant.SetPositions(plant_context, initial_positions)
        plant.SetVelocities(plant_context, initial_velocities)

    env = DrakeGymEnv(
        simulator=simulator,
        time_step=0.1,
        action_space=action_space,
        observation_space=observation_space,
        reward="reward",
        action_port_id="actions",
        observation_port_id="observations",
        reset_handler=custom_reset_handler
    )
    
    return env

def main():

    env= make_Franka_env()  

 #   simulator=make_franka_move()
    #RenderDiagram(simulator.get_system(), max_depth=1)

if __name__ == '__main__':
    main()