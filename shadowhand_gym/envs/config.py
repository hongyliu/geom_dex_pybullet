# all available objects
MESH_NAMES = ['YcbBanana', 'YcbChipsCan', 'YcbCrackerBox', 'YcbFoamBrick', 'YcbGelatinBox', 'YcbHammer',
              'YcbMasterChefCan', 'YcbMediumClamp', 'YcbMustardBottle', 'YcbPear', 'YcbPottedMeatCan', 'YcbPowerDrill',
              'YcbScissors', 'YcbStrawberry', 'YcbTennisBall', 'YcbTomatoSoupCan']
# 12 objects (avg success rate of experts = %)
ALL_TRAIN = ['YcbPear', 'YcbScissors', 'YcbHammer', 'YcbChipsCan', 'YcbCrackerBox', 'YcbFoamBrick', 'YcbGelatinBox',
             'YcbMasterChefCan', 'YcbMediumClamp', 'YcbMustardBottle']
# 4 objects (avg success rate of experts = %)
ALL_TEST = ['YcbBanana', 'YcbStrawberry', 'YcbPottedMeatCan', 'YcbPowerDrill']
ALL_SCALE = {'YcbPear': 1.0, 'YcbScissors': 0.8, 'YcbHammer': 0.7, 'YcbBanana': 1.0, 'YcbChipsCan': 0.6,
             'YcbCrackerBox': 0.6, 'YcbFoamBrick': 0.9, 'YcbGelatinBox': 0.9, 'YcbMasterChefCan': 0.8,
             'YcbMediumClamp': 1.1, 'YcbMustardBottle': 0.6, 'YcbPottedMeatCan': 0.8, 'YcbPowerDrill': 0.6,
             'YcbStrawberry': 1.0, 'YcbTomatoSoupCan': 1.0}

ALL_CLS_TRAIN = ['bathtub', 'chair', 'guitar', 'lamp', 'monitor', 'piano', 'plant', 'sink', 'table', 'toilet']

ALL_CLS_TEST = ['bathtub_test', 'chair_test', 'guitar_test', 'lamp_test', 'monitor_test', 'piano_test', 'plant_test',
                'sink_test', 'table_test', 'toilet_test']

ALL_DDPG_TRAIN = ['YcbPear', 'YcbScissors']
ALL_DDPG_TEST = ['YcbBanana']

CLS_OBJECT_LOCATION = {
    'bathtub': (0.25, 'shadowhand_gym/envs/assets/bathtub/1f5642ecc73ef347323f2769d46520fa/1f5642ecc73ef347323f2769d46520fa.urdf'),
    'bathtub_test': (0.25, 'shadowhand_gym/envs/assets/bathtub/e34833b19879020d54d7082b34825ef0/e34833b19879020d54d7082b34825ef0.urdf'),
    'chair': (0.25, 'shadowhand_gym/envs/assets/chair/1b6c268811e1724ead75d368738e0b47/1b6c268811e1724ead75d368738e0b47.urdf'),
    'chair_test': (0.25, 'shadowhand_gym/envs/assets/chair/2a8d87523e23a01d5f40874aec1ee3a6/2a8d87523e23a01d5f40874aec1ee3a6.urdf'),
    'guitar': (0.4, 'shadowhand_gym/envs/assets/guitar/2d767b3fbb8a3053b8836869016d1afd/2d767b3fbb8a3053b8836869016d1afd.urdf'),
    'guitar_test': (0.3, 'shadowhand_gym/envs/assets/guitar/a5e2f05386e4ba55a894e1aba5d3799a/a5e2f05386e4ba55a894e1aba5d3799a.urdf'),
    'lamp': (0.2, 'shadowhand_gym/envs/assets/lamp/13928/13928.urdf'),
    'lamp_test': (0.2, 'shadowhand_gym/envs/assets/lamp/14402/14402.urdf'),
    'monitor': (0.15, 'shadowhand_gym/envs/assets/monitor/3386/3386.urdf'),
    'monitor_test': (0.15, 'shadowhand_gym/envs/assets/monitor/3393/3393.urdf'),
    'piano': (0.3, 'shadowhand_gym/envs/assets/piano/d26149f3d01ffb4d9ff9560d055ab12/d26149f3d01ffb4d9ff9560d055ab12.urdf'),
    'piano_test': (0.3, 'shadowhand_gym/envs/assets/piano/f90c23de385d10f62f5fb84eadb6c469/f90c23de385d10f62f5fb84eadb6c469.urdf'),
    'plant': (0.3, 'shadowhand_gym/envs/assets/plant/1ab5af367e30508d9ec500769fff320/1ab5af367e30508d9ec500769fff320.urdf'),
    'plant_test': (0.3, 'shadowhand_gym/envs/assets/plant/1af2b0b5ca59c8d8a4136492f17b9a59/1af2b0b5ca59c8d8a4136492f17b9a59.urdf'),
    'sink': (0.15, 'shadowhand_gym/envs/assets/sink/kitchen_sink/kitchen_sink.urdf'),
    'sink_test': (0.1, 'shadowhand_gym/envs/assets/sink/sink_1/sink_1.urdf'),
    'table': (0.3, 'shadowhand_gym/envs/assets/table/1b4e6f9dd22a8c628ef9d976af675b86/1b4e6f9dd22a8c628ef9d976af675b86.urdf'),
    'table_test': (0.3, 'shadowhand_gym/envs/assets/table/783af15c06117bb29dd45a4e759f1d9c/783af15c06117bb29dd45a4e759f1d9c.urdf'),
    'toilet': (0.15, 'shadowhand_gym/envs/assets/toilet/102621/102621.urdf'),
    'toilet_test': (0.15, 'shadowhand_gym/envs/assets/toilet/102707/102707.urdf')

}
