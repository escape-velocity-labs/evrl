# import pytest
# import numpy as np
# from gym import spaces
#
# from gym_maze.envs import PathFinder5x5, PathFinder10x10, PathFinder15x15, Cell, Maze
#
#
# # Cell testing ##############################
#
# @pytest.fixture
# def cell():
#     return Cell(x=0, y=0)
#
#
# @pytest.fixture
# def open_cell():
#     cell = Cell(x=0, y=0)
#     cell.walls['S'] = False
#     return cell
#
#
# def test_new_cell_has_all_walls(cell):
#     assert all(cell.walls.values())
#
#
# def test_negative_x_raises():
#     with pytest.raises(IndexError):
#         Cell(-1, 0)
#
#
# def test_negative_y_raises():
#     with pytest.raises(IndexError):
#         Cell(0, -1)
#
#
# def test_cell_equality_works():
#     c1 = Cell(x=1, y=1)
#     c2 = Cell(x=1, y=1)
#     assert c1 == c2
#
#
# def test_different_x_breaks_cell_equality():
#     c1 = Cell(x=0, y=1)
#     c2 = Cell(x=1, y=1)
#     assert not c1 == c2
#
#
# def test_different_y_breaks_cell_equality():
#     c1 = Cell(x=1, y=0)
#     c2 = Cell(x=1, y=1)
#     assert not c1 == c2
#
#
# def test_different_walls_breaks_cell_equality():
#     c1 = Cell(x=1, y=0)
#     c1.walls['S'] = False
#     c2 = Cell(x=1, y=1)
#     assert not c1 == c2
#
#
# # Maze testing ###########################
#
#
# @pytest.fixture
# def maze():
#     return Maze(nx=5, ny=5)
#
#
# def test_invalid_nx_raises():
#     with pytest.raises(ValueError):
#         Maze(nx=2, ny=10)
#
#
# def test_invalid_ny_raises():
#     with pytest.raises(ValueError):
#         Maze(nx=10, ny=1)
#
#
# def test_correct_rows(maze):
#     assert len(maze.maze_map) == 5
#
#
# def test_correct_columns(maze):
#     assert all([len(row) == 5 for row in maze.maze_map])
#
#
# def test_cell_locations_correct(maze):
#     cell = maze.maze_map[3][2]
#     assert cell.x == 3 and cell.y == 2
#
#
# def test_maze_equality_works():
#     m1 = Maze(nx=5, ny=5)
#     m2 = Maze(nx=5, ny=5)
#     assert m1 == m2
#
#
# def test_changing_nx_breaks_maze_equality():
#     m1 = Maze(nx=5, ny=5)
#     m1.nx = 10
#     m2 = Maze(nx=5, ny=5)
#     assert not m1 == m2
#
#
# def test_changing_ny_breaks_maze_equality():
#     m1 = Maze(nx=5, ny=5)
#     m1.ny = 10
#     m2 = Maze(nx=5, ny=5)
#     assert not m1 == m2
#
#
# def test_changing_cell_breaks_maze_equality():
#     m1 = Maze(nx=5, ny=5)
#     m1.maze_map[2][3].x = 3
#     m2 = Maze(nx=5, ny=5)
#     assert not m1 == m2
#
#
# def test_cell_at(maze):
#     cell = maze.cell_at(x=2, y=3)
#     assert cell.x == 2 and cell.y == 3
#
#
# def test_cell_at_wrong_x_raises(maze):
#     with pytest.raises(IndexError):
#         maze.cell_at(x=-1, y=1)
#
#
# def test_has_all_walls(maze, cell):
#     assert Maze._has_all_walls(cell)
#
#
# def test_has_all_walls_open(maze, open_cell):
#     assert not Maze._has_all_walls(open_cell)
#
#
# def test_knock_down_wall(maze):
#     c1 = Cell(0, 0)
#     c2 = Cell(0, 1)
#     Maze._knock_down_wall(c1, c2, 'S')
#     assert not c1.walls['S'] and not c2.walls['N']
#
#
# def test_knock_down_wall_rest_stand(maze):
#     c1 = Cell(0, 0)
#     c2 = Cell(0, 1)
#     Maze._knock_down_wall(c1, c2, 'S')
#     assert all([
#         c1.walls['N'],
#         c1.walls['E'],
#         c1.walls['W'],
#         c2.walls['W'],
#         c2.walls['E'],
#         c2.walls['S']
#     ])
#
#
# # Test environment ################################
#
# @pytest.fixture
# def pathfinder_5x5():
#     return PathFinder5x5()
#
#
# @pytest.fixture
# def pathfinder_10x10():
#     return PathFinder10x10()
#
#
# @pytest.fixture
# def pathfinder_15x15():
#     return PathFinder15x15()
#
#
# def test_initial_position(pathfinder_5x5):
#     assert pathfinder_5x5.position == (0, 0)
#
#
# def test_initial_goal_location(pathfinder_5x5):
#     assert pathfinder_5x5.goal == (4, 4)
#
#
# def test_number_of_actions(pathfinder_5x5):
#     assert pathfinder_5x5.action_space.n == 4
#
#
# def test_observation_space_5x5(pathfinder_5x5):
#     assert pathfinder_5x5.observation_space == spaces.Box(np.array([0, 0]), np.array([4, 4]), dtype=int)
#
#
# def test_episode_not_done_5x5(pathfinder_5x5):
#     assert not pathfinder_5x5.done
#
#
# def test_observation_space_10x10(pathfinder_10x10):
#     assert pathfinder_10x10.observation_space == spaces.Box(np.array([0, 0]), np.array([9, 9]), dtype=int)
#
#
# def test_episode_not_done_10x10(pathfinder_10x10):
#     assert not pathfinder_10x10.done
#
#
# def test_observation_space_15x15(pathfinder_15x15):
#     assert pathfinder_15x15.observation_space == spaces.Box(np.array([0, 0]), np.array([14, 14]), dtype=int)
#
#
# def test_episode_not_done_15x15(pathfinder_15x15):
#     assert not pathfinder_15x15.done
