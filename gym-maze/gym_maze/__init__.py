from gym.envs.registration import register

register(
    id='NormalMaze-v1',
    entry_point='gym_maze.envs:NormalMaze',
)

register(
    id='RandomMaze-v1',
    entry_point='gym_maze.envs:RandomMaze',
)

register(
    id='ShapedRewardsMaze-v1',
    entry_point='gym_maze.envs:ShapedRewardsMaze',
)
