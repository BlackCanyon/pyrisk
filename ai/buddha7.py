from ai import AI
from display import CursesDisplay
import tensorflow as tf
import numpy as np
import pickle
import random
import collections
import curses
import time


terr = ["T;Alaska","T;Northwest Territories","T;Greenland","T;Alberta","T;Ontario","T;Quebec","T;Western United States","T;Eastern United States","T;Mexico","T;Venezuala","T;Peru","T;Argentina","T;Brazil","T;Iceland","T;Great Britain","T;Scandanavia","T;Western Europe","T;Northern Europe","T;Southern Europe","T;Ukraine","T;North Africa","T;Egypt","T;East Africa","T;Congo","T;South Africa","T;Madagascar","T;Middle East","T;Ural","T;Siberia","T;Yakutsk","T;Irkutsk","T;Kamchatka","T;Afghanistan","T;Mongolia","T;China","T;Japan","T;India","T;South East Asia","T;Indonesia","T;New Guinea","T;Western Australia", "T;Eastern Australia"]
atkpos = ["No Attack", "T;Afghanistan, T;China","T;Afghanistan, T;India","T;Afghanistan, T;Middle East","T;Afghanistan, T;Ukraine","T;Afghanistan, T;Ural","T;Alaska, T;Northwest Territories","T;Alaska, T;Alberta","T;Alaska, T;Kamchatka","T;Alberta, T;Alaska","T;Alberta, T;Northwest Territories","T;Alberta, T;Ontario","T;Alberta, T;Western United States","T;Argentina, T;Brazil","T;Argentina, T;Peru","T;Brazil, T;Argentina","T;Brazil, T;North Africa","T;Brazil, T;Peru","T;Brazil, T;Venezuala","T;China, T;Afghanistan","T;China, T;India","T;China, T;Mongolia","T;China, T;Siberia","T;China, T;South East Asia","T;China, T;Ural","T;Congo, T;North Africa","T;Congo, T;East Africa","T;Congo, T;South Africa","T;East Africa, T;Congo","T;East Africa, T;Egypt","T;East Africa, T;Madagascar","T;East Africa, T;Middle East","T;East Africa, T;South Africa","T;East Africa, T;North Africa","T;Eastern Australia, T;New Guinea","T;Eastern Australia, T;Western Australia","T;Eastern United States, T;Mexico","T;Eastern United States, T;Ontario","T;Eastern United States, T;Quebec","T;Eastern United States, T;Western United States","T;Egypt, T;East Africa","T;Egypt, T;Middle East","T;Egypt, T;North Africa","T;Egypt, T;Southern Europe","T;Great Britain, T;Iceland","T;Great Britain, T;Northern Europe","T;Great Britain, T;Scandanavia","T;Great Britain, T;Western Europe","T;Greenland, T;Iceland","T;Greenland, T;Northwest Territories","T;Greenland, T;Ontario","T;Greenland, T;Quebec","T;Iceland, T;Greenland","T;Iceland, T;Scandanavia","T;Iceland, T;Great Britain","T;India, T;Afghanistan","T;India, T;China","T;India, T;Middle East","T;India, T;South East Asia","T;Indonesia, T;New Guinea","T;Indonesia, T;South East Asia","T;Indonesia, T;Western Australia","T;Irkutsk, T;Kamchatka","T;Irkutsk, T;Mongolia","T;Irkutsk, T;Siberia","T;Irkutsk, T;Yakutsk","T;Japan, T;Kamchatka","T;Japan, T;Mongolia","T;Kamchatka, T;Alaska","T;Kamchatka, T;Irkutsk","T;Kamchatka, T;Mongolia","T;Kamchatka, T;Yakutsk","T;Kamchatka, T;Japan","T;Madagascar, T;East Africa","T;Madagascar, T;South Africa","T;Mexico, T;Eastern United States","T;Mexico, T;Venezuala","T;Mexico, T;Western United States","T;Middle East, T;Afghanistan","T;Middle East, T;Egypt","T;Middle East, T;India","T;Middle East, T;Southern Europe","T;Middle East, T;Ukraine","T;Middle East, T;East Africa","T;Mongolia, T;China","T;Mongolia, T;Irkutsk","T;Mongolia, T;Japan","T;Mongolia, T;Kamchatka","T;Mongolia, T;Siberia","T;New Guinea, T;Eastern Australia","T;New Guinea, T;Indonesia","T;New Guinea, T;Western Australia","T;North Africa, T;Congo","T;North Africa, T;Egypt","T;North Africa, T;Southern Europe","T;North Africa, T;Brazil","T;North Africa, T;Western Europe","T;North Africa, T;East Africa","T;Northern Europe, T;Great Britain","T;Northern Europe, T;Scandanavia","T;Northern Europe, T;Southern Europe","T;Northern Europe, T;Ukraine","T;Northern Europe, T;Western Europe","T;Northwest Territories, T;Alaska","T;Northwest Territories, T;Ontario","T;Northwest Territories, T;Alberta","T;Northwest Territories, T;Greenland","T;Ontario, T;Alberta","T;Ontario, T;Greenland","T;Ontario, T;Northwest Territories","T;Ontario, T;Quebec","T;Ontario, T;Western United States","T;Ontario, T;Eastern United States","T;Peru, T;Argentina","T;Peru, T;Brazil","T;Peru, T;Venezuala","T;Quebec, T;Eastern United States","T;Quebec, T;Greenland","T;Quebec, T;Ontario","T;Scandanavia, T;Great Britain","T;Scandanavia, T;Iceland","T;Scandanavia, T;Northern Europe","T;Scandanavia, T;Ukraine","T;Siberia, T;China","T;Siberia, T;Irkutsk","T;Siberia, T;Mongolia","T;Siberia, T;Ural","T;Siberia, T;Yakutsk","T;South Africa, T;East Africa","T;South Africa, T;Madagascar","T;South Africa, T;Congo","T;South East Asia, T;China","T;South East Asia, T;India","T;South East Asia, T;Indonesia","T;Southern Europe, T;Egypt","T;Southern Europe, T;Middle East","T;Southern Europe, T;North Africa","T;Southern Europe, T;Northern Europe","T;Southern Europe, T;Ukraine","T;Southern Europe, T;Western Europe","T;Ukraine, T;Afghanistan","T;Ukraine, T;Middle East","T;Ukraine, T;Northern Europe","T;Ukraine, T;Scandanavia","T;Ukraine, T;Southern Europe","T;Ukraine, T;Ural","T;Ural, T;Afghanistan","T;Ural, T;Siberia","T;Ural, T;Ukraine","T;Ural, T;China","T;Venezuala, T;Mexico","T;Venezuala, T;Brazil","T;Venezuala, T;Peru","T;Western Australia, T;Eastern Australia","T;Western Australia, T;New Guinea","T;Western Australia, T;Indonesia","T;Western Europe, T;Northern Europe","T;Western Europe, T;Southern Europe","T;Western Europe, T;Great Britain","T;Western Europe, T;North Africa","T;Western United States, T;Eastern United States","T;Western United States, T;Mexico","T;Western United States, T;Alberta","T;Western United States, T;Ontario","T;Yakutsk, T;Irkutsk","T;Yakutsk, T;Kamchatka","T;Yakutsk, T;Siberia"]
atkIO = [0]*167
terrIO = [0]*42
armyIO = [0]*42
a = [] #Shit, I hope I'm not still using 'a'
buddha_owned = 0 # How many countries do we have?
reward = 0
prev_owned = 0
counter = 0

BOARD_SIZE = 84

hidden_units = 120
output_units = 167
gamma = 0.99
action_log, board_log, rewards_log = [], [], []
resume = False
global probabilities
#global W2


input_positions = tf.placeholder(tf.float32, shape=[1, 84])
labels =          tf.placeholder(tf.int64)
learning_rate =   tf.placeholder(tf.float32, shape=[])
# Generate hidden layer
W1 = tf.Variable(tf.truncated_normal([BOARD_SIZE, hidden_units],
             stddev=0.1 / np.sqrt(float(BOARD_SIZE))))
b1 = tf.Variable(tf.zeros([1, hidden_units]))
h1 = tf.tanh(tf.matmul(input_positions, W1) + b1)
# Second layer -- linear classifier for action logits
W2 = tf.Variable(tf.truncated_normal([hidden_units, output_units],
             stddev=0.1 / np.sqrt(float(hidden_units))))
b2 = tf.Variable(tf.zeros([1, output_units]))
logits = tf.matmul(h1, W2) + b2
print(logits)
probabilities = tf.nn.softmax(logits)

init = tf.initialize_all_variables()
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
train_step = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cross_entropy)
# Start TF session
sess = tf.Session()
saver = tf.train.Saver()
if resume == True:
    saver.restore(sess, "/Users/John/Desktop/BlackCanyonProgramming/Risk/model.ckpt")
    print("Model restored.")
else:
    sess.run(init)

def loss_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        #running_add = running_add * gamma + r[t]
        #discounted_r[t] = running_add
        discounted_r[t] = 0
    return discounted_r

def win_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        #discounted_r[t] = len(r)
        discounted_r[t] = 1
    return discounted_r

class Buddha7AI(AI):
    """
    StupidAI: Plays a completely random game, randomly choosing and reinforcing
    territories, and attacking wherever it can without any considerations of wisdom.
    """

    def event(self, msg):
        global reward
        global prev_owned
        global ntwkIOnp
        global n_ntwkIOnp
        global board_log
        global rewards_log
        global action_log
        global conq_reward
        global counter
        TRAINING = True   # Boolean specifies training mode
        ALPHA = 0.06
        conq_reward = 0
        GameState = msg[0]
        if GameState == "victory":
            #print(len(rewards_log))
            #print(board_log)
            #print(action_log)
            #VictorCountry = msg[1]
            VictorCountry = ("%s" % msg[1])
            '''
            if VictorCountry == "P;ALPHA;Buddha7AI":
                reward = 1
                reward = reward - (turn_counter/100)
            else:
                reward = 0
            rewards_log.append(reward)
            '''
            #rewards_log = rewards_calculator(hit_log)
            rewards_log = np.array(rewards_log)
            if VictorCountry == "P;ALPHA;Buddha7AI":
                rewards_log = win_rewards(rewards_log) #If the agent wins, the reward for each action is 1.
                print('winner!')
            else:
                rewards_log = loss_rewards(rewards_log) #If the agent loses, the reward for each action is 0.

            for reward, current_board, action in zip(rewards_log, board_log, action_log):
            # Take step along gradient
                if TRAINING:
                    sess.run([train_step], #This is shameless copied from a policy gradient-based battleship AI. I'm not sure it's a good way of updating weights, but hell if I understand how to implement anything else.
                        feed_dict={input_positions:current_board, labels:[action], learning_rate:ALPHA * reward})
            counter += 1
            if counter == 100:
                save_path = saver.save(sess, "/Users/John/Desktop/BlackCanyonProgramming/Risk/model.ckpt")
                print("Model saved in file: %s" % save_path)
                counter = 0
            reward = 0
            prev_owned = 0
            #print(counter)
            action_log, board_log, rewards_log = [], [], []
            #print(VictorCountry)
        elif GameState == "claim":
            ClaimingPlayer = ("%s" % msg[1])
            ClaimedCountry = ("%s" % msg[2])
            terrNUM = terr.index(ClaimedCountry) #Returns the value in territory list that ClaimedCountry refers to.
            if ClaimingPlayer == "P;ALPHA;Buddha7AI": #Checks if the claiming player is our agent.
                terrIO[terrNUM] = 1 #Value of 1 is returned for all territories owned by our agent
                armyIO[terrNUM] += 1 #Number of armies in each territory. Stays the same whether they belong to our agent or the enemy.
            elif ClaimingPlayer != "P;ALPHA;Buddha7AI":
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
            if ConqueringPlayer == "P;ALPHA;Buddha7AI":
                terrIO[ConqNum] = 1
                conq_reward = 1
            elif ConqueringPlayer != "P;ALPHA;Buddha7AI":
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
        ntwkIOnp = np.array(networkIO)
        #print(ntwkIOnp)
        n_ntwkIOnp = np.reshape(ntwkIOnp, (1, 84))
        #print(n_ntwkIOnp)

        
    def initial_placement(self, empty, remaining):
        if empty:
            return random.choice(empty)
        else:
            t = random.choice(list(self.player.territories))
            #print(t)
            return t

    def attack(self):
        global reward
        global prev_owned
        global ntwkIOnp
        global n_ntwkIOnp
        global board_log
        global rewards_log
        global action_log
        global probabilities
        global conq_reward
        #probs = sess.run([probabilities], feed_dict={input_positions:ntwkIOnp})[0][0]
        training = True
        #final_probs = []
        #n_ntwkIOnp = tf.to_float(n_ntwkIOnp)  
        attack_probs = sess.run([probabilities], feed_dict={input_positions:n_ntwkIOnp})#{input_positions:ntwkIOnp})#[0]#[0]
        attack_probs = np.squeeze(attack_probs)
        #print(attack_probs)
        #validMove_prob = attack_probs[0]
        #print(attack_probs[0])
        #final_probs.append(validMove_prob)
        #atk_choices = ['No Attack'] #Remember to put "no attack" back in at some point
        #validMove_index = [0] #Remember to put zero in for "no attack" command
        #rewards_log = []
        #t_list = [0]
        #adj_list = [0]
        #action_log = []
        can_attack = True
        #board_log = []
        board_log.append(n_ntwkIOnp)
        while can_attack:
            final_probs = []
            validMove_prob = attack_probs[0]
            final_probs.append(validMove_prob)
            atk_choices = ['No Attack'] #Remember to put "no attack" back in at some point
            validMove_index = [0]
            t_list = [0]
            adj_list = [0]
            for t in self.player.territories:
                if t.forces > 1:
                    for adj in t.adjacent(friendly=False):
                        #Not sure if this is important
                        #n_ntwkIOnp = tf.to_float(n_ntwkIOnp)
                        #print(tf.shape(n_ntwkIOnp))    
                        #attack_probs = sess.run([probabilities], feed_dict={input_positions:n_ntwkIOnp})#[0][0]
                        #let's see
                        t_list.append(t) #Creates a list of territories to attack from
                        adj_list.append(adj) #Creates a list of territories to attack
                        AtkStr = ("%s, %s" % (t, adj)) #Creates a string to compare to atkpos
                        atk_choices.append(AtkStr) #Compiles all possible attack choices
                        validMove = atkpos.index(AtkStr) #compares output to atkpos to find legal moves
                        validMove_index.append(validMove)
                        validMove_prob = attack_probs[validMove] # Finds the associated probability
                        final_probs.append(validMove_prob) #Adds the probability to a list

            final_probs = [p / sum(final_probs) for p in final_probs]
            #print(final_probs)

            if training == True:
                    #print(len(atk_choices))
                    #print(len(final_probs))   
                atk_index = np.random.choice(len(atk_choices), p=final_probs)
                #print(final_probs)
                chosen_move = validMove_index[atk_index]
                action_log.append(chosen_move)
                    #print(atk_index)           
            else:
                atk_index = np.argmax(final_probs)

            if atk_index == 0:
                board_log.append(n_ntwkIOnp)
                can_attack = False
                reward = 1
                rewards_log.append(reward)
            else:
                #print(t_list[atk_index], adj_list[atk_index])
                board_log.append(n_ntwkIOnp)
                yield (t_list[atk_index], adj_list[atk_index], None, None)
                reward = 1
                #print(reward)
                rewards_log.append(reward)

        #reward = sum(terrIO)
        #reward = reward - prev_owned
        #print(reward)
        #prev_owned = sum(terrIO)
        #board_log.append(ntwkIOnp)
        #rewards_log.append(reward)
        #print(reward)


    def reinforce(self, available):
        border = [t for t in self.player.territories if t.border]
        result = collections.defaultdict(int)
        for i in range(available):
            t = random.choice(border)
            result[t] += 1
        return result
