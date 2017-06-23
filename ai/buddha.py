from ai import AI
from display import CursesDisplay
import numpy as np
import pickle
import random
import collections
import curses
import time


terr = ["T;Alaska","T;Northwest Territories","T;Greenland","T;Alberta","T;Ontario","T;Quebec","T;Western United States","T;Eastern United States","T;Mexico","T;Venezuala","T;Peru","T;Argentina","T;Brazil","T;Iceland","T;Great Britain","T;Scandanavia","T;Western Europe","T;Northern Europe","T;Southern Europe","T;Ukraine","T;North Africa","T;Egypt","T;East Africa","T;Congo","T;South Africa","T;Madagascar","T;Middle East","T;Ural","T;Siberia","T;Yakutsk","T;Irkutsk","T;Kamchatka","T;Afghanistan","T;Mongolia","T;China","T;Japan","T;India","T;South East Asia","T;Indonesia","T;New Guinea","T;Western Australia", "T;Eastern Australia"]
atkpos = ["No Attack", "T;Afghanistan, T;China","T;Afghanistan, T;India","T;Afghanistan, T;Middle East","T;Afghanistan, T;Ukraine","T;Afghanistan, T;Ural","T;Alaska, T;Northwest Territories","T;Alaska, T;Alberta","T;Alaska, T;Kamchatka","T;Alberta, T;Alaska","T;Alberta, T;Northwest Territories","T;Alberta, T;Ontario","T;Alberta, T;Western United States","T;Argentina, T;Brazil","T;Argentina, T;Peru","T;Brazil, T;Argentina","T;Brazil, T;North Africa","T;Brazil, T;Peru","T;Brazil, T;Venezuala","T;China, T;Afghanistan","T;China, T;India","T;China, T;Mongolia","T;China, T;Siberia","T;China, T;South East Asia","T;China, T;Ural","T;Congo, T;North Africa","T;Congo, T;East Africa","T;Congo, T;South Africa","T;East Africa, T;Congo","T;East Africa, T;Egypt","T;East Africa, T;Madagascar","T;East Africa, T;Middle East","T;East Africa, T;South Africa","T;East Africa, T;North Africa","T;Eastern Australia, T;New Guinea","T;Eastern Australia, T;Western Australia","T;Eastern United States, T;Mexico","T;Eastern United States, T;Ontario","T;Eastern United States, T;Quebec","T;Eastern United States, T;Western United States","T;Egypt, T;East Africa","T;Egypt, T;Middle East","T;Egypt, T;North Africa","T;Egypt, T;Southern Europe","T;Great Britain, T;Iceland","T;Great Britain, T;Northern Europe","T;Great Britain, T;Scandanavia","T;Great Britain, T;Western Europe","T;Greenland, T;Iceland","T;Greenland, T;Northwest Territories","T;Greenland, T;Ontario","T;Greenland, T;Quebec","T;Iceland, T;Greenland","T;Iceland, T;Scandanavia","T;Iceland, T;Great Britain","T;India, T;Afghanistan","T;India, T;China","T;India, T;Middle East","T;India, T;South East Asia","T;Indonesia, T;New Guinea","T;Indonesia, T;South East Asia","T;Indonesia, T;Western Australia","T;Irkutsk, T;Kamchatka","T;Irkutsk, T;Mongolia","T;Irkutsk, T;Siberia","T;Irkutsk, T;Yakutsk","T;Japan, T;Kamchatka","T;Japan, T;Mongolia","T;Kamchatka, T;Alaska","T;Kamchatka, T;Irkutsk","T;Kamchatka, T;Mongolia","T;Kamchatka, T;Yakutsk","T;Kamchatka, T;Japan","T;Madagascar, T;East Africa","T;Madagascar, T;South Africa","T;Mexico, T;Eastern United States","T;Mexico, T;Venezuala","T;Mexico, T;Western United States","T;Middle East, T;Afghanistan","T;Middle East, T;Egypt","T;Middle East, T;India","T;Middle East, T;Southern Europe","T;Middle East, T;Ukraine","T;Middle East, T;East Africa","T;Mongolia, T;China","T;Mongolia, T;Irkutsk","T;Mongolia, T;Japan","T;Mongolia, T;Kamchatka","T;Mongolia, T;Siberia","T;New Guinea, T;Eastern Australia","T;New Guinea, T;Indonesia","T;New Guinea, T;Western Australia","T;North Africa, T;Congo","T;North Africa, T;Egypt","T;North Africa, T;Southern Europe","T;North Africa, T;Brazil","T;North Africa, T;Western Europe","T;North Africa, T;East Africa","T;Northern Europe, T;Great Britain","T;Northern Europe, T;Scandanavia","T;Northern Europe, T;Southern Europe","T;Northern Europe, T;Ukraine","T;Northern Europe, T;Western Europe","T;Northwest Territories, T;Alaska","T;Northwest Territories, T;Ontario","T;Northwest Territories, T;Alberta","T;Northwest Territories, T;Greenland","T;Ontario, T;Alberta","T;Ontario, T;Greenland","T;Ontario, T;Northwest Territories","T;Ontario, T;Quebec","T;Ontario, T;Western United States","T;Ontario, T;Eastern United States","T;Peru, T;Argentina","T;Peru, T;Brazil","T;Peru, T;Venezuala","T;Quebec, T;Eastern United States","T;Quebec, T;Greenland","T;Quebec, T;Ontario","T;Scandanavia, T;Great Britain","T;Scandanavia, T;Iceland","T;Scandanavia, T;Northern Europe","T;Scandanavia, T;Ukraine","T;Siberia, T;China","T;Siberia, T;Irkutsk","T;Siberia, T;Mongolia","T;Siberia, T;Ural","T;Siberia, T;Yakutsk","T;South Africa, T;East Africa","T;South Africa, T;Madagascar","T;South Africa, T;Congo","T;South East Asia, T;China","T;South East Asia, T;India","T;South East Asia, T;Indonesia","T;Southern Europe, T;Egypt","T;Southern Europe, T;Middle East","T;Southern Europe, T;North Africa","T;Southern Europe, T;Northern Europe","T;Southern Europe, T;Ukraine","T;Southern Europe, T;Western Europe","T;Ukraine, T;Afghanistan","T;Ukraine, T;Middle East","T;Ukraine, T;Northern Europe","T;Ukraine, T;Scandanavia","T;Ukraine, T;Southern Europe","T;Ukraine, T;Ural","T;Ural, T;Afghanistan","T;Ural, T;Siberia","T;Ural, T;Ukraine","T;Ural, T;China","T;Venezuala, T;Mexico","T;Venezuala, T;Brazil","T;Venezuala, T;Peru","T;Western Australia, T;Eastern Australia","T;Western Australia, T;New Guinea","T;Western Australia, T;Indonesia","T;Western Europe, T;Northern Europe","T;Western Europe, T;Southern Europe","T;Western Europe, T;Great Britain","T;Western Europe, T;North Africa","T;Western United States, T;Eastern United States","T;Western United States, T;Mexico","T;Western United States, T;Alberta","T;Western United States, T;Ontario","T;Yakutsk, T;Irkutsk","T;Yakutsk, T;Kamchatka","T;Yakutsk, T;Siberia"]
atkIO = [0]*166
terrIO = [0]*42
armyIO = [0]*42
a = [] #Shit, I hope I'm not still using 'a'
buddha_owned = 0 # How many countries do we have?
reward = 0
prev_owned = 0
counter = 0

# hyperparameters
episode_number = 0
batch_size = 10
gamma = 0.99 # discount factor for reward
decay_rate = 0.99
num_hidden_layer_neurons = 200
num_output_neurons = 167
input_dimensions = 84 * 1
learning_rate = 1e-4
episode_number = 0
reward_sum = 0
running_reward = None
resume = True
#episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

if resume:
  weights = pickle.load(open('save.p', 'rb'))
  print('yay')
else:
    weights = {}
    weights['W1'] = np.random.randn(input_dimensions, num_hidden_layer_neurons) / np.sqrt(input_dimensions),
    weights['W2'] = np.random.randn(num_hidden_layer_neurons, num_output_neurons) / np.sqrt(num_hidden_layer_neurons)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(vector):
    vector[vector < 0] = 0
    return vector

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def apply_neural_nets(ntwkIOnp, weights):
    global attack_probs
    global hidden_layer_values
    hidden_layer_values = np.dot(ntwkIOnp, weights['W1']) #+ weights['B1']
    hidden_layer_values = relu(hidden_layer_values)
    output_neuron_values = np.dot(hidden_layer_values, weights['W2']) #+ weights['B2']
    attack_probs = softmax(output_neuron_values)
    return attack_probs #returns both to the event function and becomes a global variable. It's an awful
    #solution, but I'm not good at programming and it was the easiest way to move past that
    #particular issue.


class BuddhaAI(AI):
    """
    StupidAI: Plays a completely random game, randomly choosing and reinforcing
    territories, and attacking wherever it can without any considerations of wisdom.
    """

    def event(self, msg):
        global ntwkIOnp
        global counter
        GameState = msg[0]
        if GameState == "victory":
            VictorCountry = msg[1]
            counter += 1
            pickle.dump(weights, open('save.p', 'wb'))
            #print(VictorCountry)
        elif GameState == "claim":
            ClaimingPlayer = ("%s" % msg[1])
            ClaimedCountry = ("%s" % msg[2])
            terrNUM = terr.index(ClaimedCountry) #Returns the value in territory list that ClaimedCountry refers to.
            if ClaimingPlayer == "P;ALPHA;BuddhaAI": #Checks if the claiming player is our agent.
                terrIO[terrNUM] = 1 #Value of 1 is returned for all territories owned by our agent
                armyIO[terrNUM] += 1 #Number of armies in each territory. Stays the same whether they belong to our agent or the enemy.
            elif ClaimingPlayer != "P;ALPHA;BuddhaAI":
                terrIO[terrNUM] = 0 #Value of 0 is returned for all other territories
                armyIO[terrNUM] += 1 #Number of armies in each territory
        elif GameState == "reinforce":
            ReinforcedCountry = ("%s" % msg[2])
            ReinforcedNum = (msg[3])
            reinNUM = terr.index(ReinforcedCountry)
            armyIO[reinNUM] += ReinforcedNum
        elif GameState == "conquer":
            ConqueringPlayer = ("%s" % msg[1])
            AttackingCountry = ("%s" % msg[3])
            ConqueredCountry = ("%s" % msg[4])
            ArmyCount = msg[6] #Returns an (x,y) value of armies in AttackingCountry and ConqueredCountry
            AttackNum = terr.index(AttackingCountry)
            ConqNum = terr.index(ConqueredCountry) #Returns the value in territory list that ConqueredCountry refers to.
            armyIO[AttackNum] = ArmyCount[0]
            armyIO[ConqNum] = ArmyCount[1]
            if ConqueringPlayer == "P;ALPHA;BuddhaAI":
                terrIO[ConqNum] = 1
            elif ConqueringPlayer != "P;ALPHA;BuddhaAI":
                terrIO[ConqNum] = 0
        elif GameState == "defeat":
            FailingCountry = ("%s" % msg[3])
            DefendingCountry = ("%s" % msg[4])
            ArmyCount = msg[6] #Returns an (x,y) value of armies in FailingCountry and DefendingCountry
            FailNum = terr.index(FailingCountry)
            DefNum = terr.index(DefendingCountry) #Returns the value in territory list that Defending Country refers to.
            armyIO[FailNum] = ArmyCount[0]
            armyIO[DefNum] = ArmyCount[1]
        TotalArmies = sum(armyIO)
        if TotalArmies != 0:
            ArmyNormal = [x / TotalArmies for x in armyIO]
        elif TotalArmies == 0:
            ArmyNormal = armyIO
        networkIO = terrIO + ArmyNormal
        ntwkIOnp = np.array(networkIO) # This is the important bit of all of that code. It returns an 84x1 array of the current
                                        # board state. 42 values for country ownership (us=1, not us=0) and 42 values for number
                                        # of armies (represented as a percentage of total armies).

        output_probs = apply_neural_nets(ntwkIOnp, weights)
        
    def initial_placement(self, empty, remaining):
        if empty:
            return random.choice(empty)
        else:
            t = random.choice(list(self.player.territories))
            return t

    def attack(self):
        global reward
        global prev_owned
        global ntwkIOnp
        global action_log
        global attack_probs
        #global episode_hidden_layer_values
        training = True
        final_probs = []
        validMove_prob = attack_probs[0, 1]
        final_probs.append(validMove_prob)
        atk_choices = ['No Attack']
        validMove_index = [0]
        reward_log = []
        t_list = [0]
        adj_list = [0]
        can_attack = True

        while can_attack:
            for t in self.player.territories:
                if t.forces > 1:
                    for adj in t.adjacent(friendly=False):
                        t_list.append(t) #Creates a list of territories to attack from
                        adj_list.append(adj) #Creates a list of territories to attack
                        AtkStr = ("%s, %s" % (t, adj)) #Creates a string to compare to atkpos
                        atk_choices.append(AtkStr) #Compiles all possible attack choices
                        validMove = atkpos.index(AtkStr) #compares output to atkpos to find legal moves
                        validMove_index.append(validMove)
                        validMove_prob = attack_probs[0, validMove] # Finds the associated probability
                        final_probs.append(validMove_prob) #Adds the probability to a list

            final_probs = [p / sum(final_probs) for p in final_probs]

            if training == True:
                atk_index = np.random.choice(len(atk_choices), p=final_probs)
                chosen_move = validMove_index[atk_index]
                #print(atk_index)            
            else:
                atk_index = np.argmax(final_probs)

            if atk_index == 0:
                can_attack = False
            else:
                yield (t_list[atk_index], adj_list[atk_index], None, None)
        reward = sum(terrIO)
        reward = reward - prev_owned
        prev_owned = sum(terrIO)
        #print(reward)


    def reinforce(self, available):
        border = [t for t in self.player.territories if t.border]
        result = collections.defaultdict(int)
        for i in range(available):
            t = random.choice(border)
            result[t] += 1
        return result
