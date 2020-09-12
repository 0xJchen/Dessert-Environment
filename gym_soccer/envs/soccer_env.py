import gym
from gym import spaces
import numpy as np
from tqdm import tqdm

DIE_REWARD = -100000
START = 0; VILLAGE = 1; MINE = 2; END = 3
INVALID = -1
SUNNY = 0; HOT = 1; STORMY = 2

class Level1(gym.Env):
  def __init__(self):
    super(Level1, self).__init__()
    # actions: 0: to village 1: to mine 2: to end 3: mining 4: buying
    self.action_space = spaces.Discrete(5)
    # observations [location, time, water1, food1, water2, food2, money]
    # location 0: start 1: village 2: mine 3: end
    self.observation_space = spaces.MultiDiscrete([4, 30, 400, 600, 400, 600, 15000])
    self.state = np.array([0,0,0,0,0,0,0])

    # 0: sunny 1: hot 2: stormy
    self.weather = [1,1,0,2,0,1,2,0,1,1, 2,1,0,1,1,1,2,2,1,1, 0,0,1,0,2,1,0,0,1,1]
    self.cost_water = [5,8,10]
    self.cost_food = [7,6,10]
    self.price_water = 5
    self.price_food = 10
    self.quantity_water = 5
    self.quantity_food = 2

    # 0: start 1: village 2: mine 3: end
    self.distance = [[0,6,8,3],[6,0,2,3],[8,2,0,5],[3,3,5,0]]
    self.move_cost = np.zeros((30,4,4,2))
    self.move_time = np.zeros((30,4,4))

    # initialization
    self._get_move_info()
    self.reset()

  def _get_move_info(self):
    for t in range(30):
      for i in range(4):
        weather = self.weather[t]
        self.move_cost[t][i][i][0] = self.cost_water[weather]
        self.move_cost[t][i][i][1] = self.cost_food[weather]
        for j in range(i+1,4):
          distance = self.distance[i][j]
          period = 0
          while distance > 0:
            if t + period >= 30:
              self.move_cost[t][i][j][0] = self.move_cost[t][j][i][0] = \
              self.move_cost[t][i][j][1] = self.move_cost[t][j][i][1] = \
              self.move_time[t][i][j] = self.move_time[t][j][i] = INVALID
              break
            else:
              weather = self.weather[t + period]
              # stormy
              if weather == STORMY:
                self.move_cost[t][i][j][0] += self.cost_water[weather]
                self.move_cost[t][i][j][1] += self.cost_food[weather]
              else:
                self.move_cost[t][i][j][0] += 2 * self.cost_water[weather]
                self.move_cost[t][i][j][1] += 2 * self.cost_food[weather]
                distance -= 1
              period += 1
          else:
            self.move_cost[t][j][i][0] = self.move_cost[t][i][j][0]
            self.move_cost[t][j][i][1] = self.move_cost[t][i][j][1]
            self.move_time[t][j][i] = self.move_time[t][i][j] = period

  def step(self, action):
    location = int(self.state[0])
    day = int(self.state[1])
    water1 = int(self.state[2])
    food1 = int(self.state[3])
    water2 = int(self.state[4])
    food2 = int(self.state[5])
    money = self.state[6]

    reward = 0
    done = False
    info = dict()
    # move to 1:village
    if action == 0:
      water_cost = self.move_cost[day][location][VILLAGE][0]
      food_cost = self.move_cost[day][location][VILLAGE][1]
      period = self.move_time[day][location][VILLAGE]
      if water_cost == INVALID or food_cost == INVALID or period == INVALID or \
         water_cost > water1 + water2 or food_cost > food1 + food2 or day + period >= 30:
        reward = DIE_REWARD
        done = True
      else:
        location = VILLAGE
        day += period
        if water_cost > water1:
          water1 = 0
          water2 -= (water_cost - water1)
        else:
          water1 -= water_cost
        if food_cost > food1:
          food1 = 0
          food2 -= (food_cost - food1)
        else:
          food1 -= food_cost
        reward = - water_cost * self.price_water - food_cost * self.price_food
    # move to 2:mine
    elif action == 1:
      water_cost = self.move_cost[day][location][MINE][0]
      food_cost = self.move_cost[day][location][MINE][1]
      period = self.move_time[day][location][MINE]
      if water_cost == INVALID or food_cost == INVALID or period == INVALID or \
         water_cost > water1 + water2 or food_cost > food1 + food2 or day + period >= 30:
        reward = DIE_REWARD
        done = True
      else:
        location = MINE
        day += period
        if water_cost > water1:
          water1 = 0
          water2 -= (water_cost - water1)
        else:
          water1 -= water_cost
        if food_cost > food1:
          food1 = 0
          food2 -= (food_cost - food1)
        else:
          food1 -= food_cost
        reward = - water_cost * self.price_water - food_cost * self.price_food
    # move to 3:end
    elif action == 2:
      water_cost = self.move_cost[day][location][END][0]
      food_cost = self.move_cost[day][location][END][1]
      period = self.move_time[day][location][END]
      done = True
      if water_cost == INVALID or food_cost == INVALID or period == INVALID or \
         water_cost > water1 + water2 or food_cost > food1 + food2 or day + period >= 30:
        reward = DIE_REWARD
      else:
        location = END
        day += period
        if water_cost > water1:
          water1 = 0
          water2 -= (water_cost - water1)
        else:
          water1 -= water_cost
        if food_cost > food1:
          food1 = 0
          food2 -= (food_cost - food1)
        else:
          food1 -= food_cost
        money += (self.price_water * (water1 + 2 * water2) + self.price_food * (food1 + 2 * food2))
        reward = - water_cost * self.price_water - food_cost * self.price_food + \
            (self.price_water * (water1 + 2 * water2) + self.price_food * (food1 + 2 * food2))
    # mining
    elif action == 3:
      if location == MINE:
        water_cost = 3 * self.move_cost[day][MINE][MINE][0]
        food_cost = 3 * self.move_cost[day][MINE][MINE][1]
        if water_cost == INVALID or food_cost == INVALID or \
           water_cost > water1 + water2 or food_cost > food1 + food2 or day + 1 >= 30:
          reward = DIE_REWARD
          done = True
        else:
          day += 1
          if water_cost > water1:
            water1 = 0
            water2 -= (water_cost - water1)
          else:
            water1 -= water_cost
          if food_cost > food1:
            food1 = 0
            food2 -= (food_cost - food1)
          else:
            food1 -= food_cost
          reward = - water_cost * self.price_water - food_cost * self.price_food + 1000
          money += 1000
      else:
        reward = DIE_REWARD
        done = True

    # buying
    elif action == 4:
      if location == VILLAGE:
        unoccupied = 1200 - self.quantity_water * (water1 + water2) - self.quantity_food * (food1 + food2)
        amount = unoccupied // (self.quantity_water + self.quantity_food)
        water2 += amount
        food2 += amount
        reward = 0
        money -= (amount * self.price_water + amount * self.price_food)
      else:
        reward = DIE_REWARD
        done = True


    self.state = np.array([location, day, water1, food1, water2, food2, money])

    return self.state, reward, done, info

  def reset(self, init_water=200, init_food=300):
    self.state = np.array([0, 0, init_water, init_food, 0, 0, 10000 - init_water * self.price_water - init_food * self.price_food])
    return self.state

  def render(self):
    if self.state[0] == 0:
      print('Day {day:<{len}} Location: Start'.format(day=self.state[1], len=10))
    elif self.state[0] == 1:
      print('Day {day:<{len}} Location: Village'.format(day=self.state[1], len=10))
    elif self.state[0] == 2:
      print('Day {day:<{len}} Location: Mine'.format(day=self.state[1], len=10))
    elif self.state[0] == 3:
      print('Day {day:<{len}} Location: End'.format(day=self.state[1], len=10))
    print('Initial water: {water1:<{len}} bought water: {water2:<{len}}'.format(water1=self.state[2], water2=self.state[4],len=10))
    print('Initial food:  {food1:<{len}} bought food: {food2:<{len}}'.format(food1=self.state[3], food2=self.state[5], len=10))
    print('Money: ', self.state[6])
    print()


# actions: 0: to village 1: to mine 2: to end 3: mining 4: buying
a_village = 0; a_mine = 1; a_end = 2; a_mining = 3; a_buying = 4
env = Level1()
env.reset()
env.render()

actions = [
  a_village, a_mine, a_mining, a_mining, a_mining, a_mining
]

# for action in actions:
#   obs, reward, done, _ = env.step(action)
#   if done:
#     print('die')
#     break
#   env.render()#