import gym
from gym import spaces
import numpy as np
import random
from queue import PriorityQueue
from queue import Queue

class Client:
    def __init__(self, client_id, inquiry_type, priority_class, arrival_time, service_expected_start_time=0):
        self.client_id = client_id
        self.inquiry_type = inquiry_type  # 'basic' or 'complex'
        self.priority_class = priority_class  # 'normal', 'VIP', etc.
        self.arrival_time = arrival_time
        self.service_expected_start_time = service_expected_start_time
        self.service_start_time = 0
        self.service_expected_end_time = 0
        self.service_end_time = 0
        self.abandonment_time = 0
        self.server_id = None
    
    def assign_server(self, server_id):
        self.server_id = server_id

class Staff:
    def __init__(self, staff_id):
        self.staff_id = staff_id
        self.available = True
        self.current_client = None
        self.num_clients_served = 0
        self.num_abandon_client = 0
        self.time_serving_client = 0
        self.staff_queue = Queue()
        self.working_time = 0
        self.last_idle_time = 0
        self.idle_time = 0

class Staff_Pool:
    def __init__(self, staff_pool_size):
        self.staff_list = [Staff(i) for i in range(staff_pool_size)]

    def num_available_staff(self):
        return len([staff for staff in self.staff_list if staff.available])
    
    def assign_client(self, action, client):
        if action < len(self.staff_list) and self.staff_list[action].available:
            self.staff_list[action].available = False
            self.staff_list[action].current_client = client
    
    def available_staff(self):
        return [0 if staff.available else 1 for staff in self.staff_list]

class Event:
    def __init__(self, client_id, time, event_id):
        self.client_id = client_id
        self.time = time
        self.event_id = event_id

class Arrival(Event):
    def __init__(self, client_id, time, enquiry_type, event_id):
        super().__init__(client_id, time, event_id)
        # self.client = client
        self.type = enquiry_type
    
    def __str__(self):
        return f"Arrival T{self.type}"

class Departure(Event):
    def __init__(self, client_id, time, serverid, event_id):
        super().__init__(client_id, time, event_id)
        self.serverid = serverid
    
    def __str__(self):
        return f"Departure"

class Abandonment(Event):
    def __init__(self, client_id, time, serverid, event_id):
        super().__init__(client_id, time, event_id)
        self.serverid = serverid
        
    def __str__(self):
        return f"Abandonment"


# The Fundenamental Call Centre Environment
class CallCentreEnv(gym.Env):
    def __init__(self, staff_pool_size = 2, time_until=28800, arrival_rate=[100,120], service_rate=[[120, 190], [150,170]], abandonment_rate=[300, 400], max_staff_queue= 14, random_run=True):
        super(CallCentreEnv, self).__init__()
        self.staff_pool_size = staff_pool_size
        self.time_until = time_until
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.abandonment_rate = abandonment_rate
        self.max_staff_queue = max_staff_queue
        self.vip_waiting_time = 300
        self.normal_waiting_time = 600
        
        self.staff_pool = Staff_Pool(self.staff_pool_size)
        self.event_queue = PriorityQueue()
        self.time = 0
        self.client_counter = 0
        self.served_clients_counter = 0
        self.abandonment_counter = 0
        self.event_counter = 0
        self.served_client = []
        self.abandonment_client = []
        self.happened_event = []
        self.waiting_time_list = []
        self.idle_time_list = []
        self.idle_time = 0
        
        
        self.action_space = spaces.Discrete(self.staff_pool_size)
        self.observation_space = spaces.Box(low=0, high=self.max_staff_queue, shape=(self.staff_pool_size + 1,), dtype=np.float32)

        # Initialize the first arrival event
        init_time = 0
        for i, rate in enumerate(self.arrival_rate):
            self.event_counter += 1
            self.event_queue.put((init_time, Arrival(0, init_time, i, self.event_counter)))
            init_time += 0.0001
        
        if not random_run:
            np.random.seed(1234)
            random.seed(1234)
    
    def reset(self):
        self.event_queue = PriorityQueue()
        self.staff_pool = Staff_Pool(self.staff_pool_size)
        self.time = 0
        self.client_counter = 0
        self.served_clients_counter = 0
        self.abandonment_counter = 0
        self.event_counter = 0
        self.avg_waiting_time = 0
        self.served_client = []
        self.abandonment_client = []
        self.happened_event = []
        self.waiting_time_list = []
        self.idle_time_list = []
        self.idle_time = 0
        
        # Initialize the first arrival event
        init_time = 0
        for i, rate in enumerate(self.arrival_rate):
            self.event_counter += 1
            self.event_queue.put((init_time, Arrival(0, init_time, i, self.event_counter)))
            init_time += 0.0001

        return self._get_observation() 


    def _get_observation(self):
        obs = np.zeros((self.staff_pool_size +1,), dtype=np.float32)

        for i, staff in enumerate(self.staff_pool.staff_list):
            obs[i] = staff.staff_queue.qsize()
        
        if not self.event_queue.empty():
            _, event = self.event_queue.queue[0]
            obs[-1] = event.type
        else:
            obs[-1] = 0

        return obs
    
    
    def step(self, action):
        reward = 0
        valid_actions = self._available_actions()
        index, event = self.event_queue.get()
        

        if action not in valid_actions:
            reward -= 125
        
        selected_staff = self.staff_pool.staff_list[action]
        num_idle_staff = sum(staff.available for staff in self.staff_pool.staff_list)

        waiting_clients = [client for staff in self.staff_pool.staff_list for client in staff.staff_queue.queue if not staff.staff_queue.empty()]
        num_waiting_clients = len(waiting_clients)
        

        if isinstance(event, Arrival):
            self.time = event.time
            self.client_counter += 1

            new_client = self._client_generator(self.client_counter, event)
            new_client.assign_server(action)
            self.staff_pool.staff_list[action].staff_queue.put(new_client)
            
            if self.time < self.time_until:
                next_arrival_time = self.time + np.random.poisson(self.arrival_rate[event.type])
                if next_arrival_time < self.time_until: 
                    self.event_counter += 1
                    self.event_queue.put((next_arrival_time, Arrival(new_client.client_id+1, next_arrival_time, event.type, self.event_counter)))
            
            self.event_counter += 1
            abandonment_time = self.time + np.random.exponential(self.abandonment_rate[event.type])
            self.event_queue.put((abandonment_time, Abandonment(new_client.client_id, abandonment_time, action, self.event_counter)))

            if self.staff_pool.staff_list[action].available and not self.staff_pool.staff_list[action].staff_queue.empty():
                reward += self._move_client_to_staff(action)
                    
        self.avg_waiting_time = 0
        if num_waiting_clients > 0:
            average_waiting_time = sum(self.time - client.arrival_time for client in waiting_clients) / num_waiting_clients
            self.avg_waiting_time = average_waiting_time
            reward -= average_waiting_time  # Adjust reward based on average waiting time
        
        self.waiting_time_list.append(self.avg_waiting_time)
        self.happened_event.append([event, [staff.staff_queue.qsize() for staff in self.staff_pool.staff_list], [[i,str(j)] for i,j in self.event_queue.queue]])
        
        idle_time = 0
        if num_idle_staff > 0:
            for staff in self.staff_pool.staff_list:
                if staff.available:
                    idle_time += self.time - staff.last_idle_time
                    staff.idle_time += self.time - staff.last_idle_time
        
        self.idle_time_list.append(idle_time)
        self.idle_time = idle_time

        reward -= idle_time 
        reward += self._process_background_events()
        
        terminated = self.event_queue.empty()

        return self._get_observation(), reward, terminated, {}

    def _remove_abandonment_by_id(self, target_event_id: int) -> None:
        temp_list = []
        while not self.event_queue.empty():
            priority, event = self.event_queue.get()
            if not (isinstance(event, Abandonment) and event.client_id == target_event_id):
                temp_list.append((priority, event))
        for item in temp_list:
            self.event_queue.put(item)

    def _move_client_to_staff(self, staff_id):
        client = self.staff_pool.staff_list[staff_id].staff_queue.get()
        client.service_start_time = self.time
        
        self._remove_abandonment_by_id(client.client_id)
        
        self.staff_pool.staff_list[staff_id].available = False
        self.staff_pool.staff_list[staff_id].current_client = client

        departure_time = client.service_start_time + np.random.exponential(self.service_rate[staff_id][client.inquiry_type])
        self.event_counter += 1
        self.event_queue.put((departure_time, Departure(client.client_id, departure_time, staff_id, self.event_counter)))

        if client.service_start_time < client.service_expected_start_time:
            reward = 0
        else:
            reward = -0
        
        return reward


    def _available_actions(self):
        
        possible_action = [i for i, staff in enumerate(self.staff_pool.staff_list) if staff.staff_queue.qsize() < self.max_staff_queue]
        
        return possible_action

    def _process_background_events(self, accumulated_reward=0):
  
        if self.event_queue.empty():
            return accumulated_reward  # No event to process

        _, event = self.event_queue.queue[0]  # Peek at the next event

        if isinstance(event, Abandonment):
            # Process abandonment immediately
            _, abandonment_event = self.event_queue.get()  # Remove event
            self.happened_event.append([abandonment_event, [staff.staff_queue.qsize() for staff in self.staff_pool.staff_list], [[i,str(j)] for i,j in self.event_queue.queue]])
            self.time = event.time
            self.abandonment_counter += 1
            
            staff = self.staff_pool.staff_list[event.serverid]
            staff.num_abandon_client += 1
            temp_queue = Queue()
            
            while not staff.staff_queue.empty():
                client = staff.staff_queue.get()
                if client.client_id != event.client_id:
                    temp_queue.put(client)
                else:
                    client.abandonment_time = self.time
                    self.abandonment_client.append (client)

            staff.staff_queue = temp_queue 

            return self._process_background_events(accumulated_reward - 125) 
 

        elif isinstance(event, Departure):
            # Process Departure if no clients are waiting
            _, departure_event = self.event_queue.get()  # Remove event
            self.happened_event.append([departure_event, [staff.staff_queue.qsize() for staff in self.staff_pool.staff_list], [[i,str(j)] for i,j in self.event_queue.queue]])
            self.time = event.time
            self.served_clients_counter += 1
            self.staff_pool.staff_list[event.serverid].num_clients_served += 1

            self.staff_pool.staff_list[event.serverid].available = True
            self.staff_pool.staff_list[event.serverid].last_idle_time = self.time
            self.staff_pool.staff_list[event.serverid].current_client.service_end_time = self.time
            self.staff_pool.staff_list[event.serverid].working_time += self.staff_pool.staff_list[event.serverid].current_client.service_end_time - self.staff_pool.staff_list[event.serverid].current_client.service_start_time
            self.served_client.append(self.staff_pool.staff_list[event.serverid].current_client)
            self.staff_pool.staff_list[event.serverid].current_client = None

            if self.staff_pool.staff_list[event.serverid].staff_queue.qsize() > 0:
                reward = self._move_client_to_staff(event.serverid)
                self.happened_event[-1][-1] = [[i,str(j)] for i,j in self.event_queue.queue]
                return self._process_background_events(accumulated_reward + reward)

            return self._process_background_events(accumulated_reward)  # Recursively process next event
        
        else:
            # If the next event is NOT a background event, stop processing
            return accumulated_reward

    def _client_generator(self, id, event):
        # Randomly generate client class
        inquiry_type = event.type
        priority_class = random.choice(['normal', 'VIP'])
        expected_waiting_time = self.vip_waiting_time if priority_class == 'VIP' else self.normal_waiting_time
        return Client(id, inquiry_type, priority_class, event.time, event.time + expected_waiting_time)

    def _system_statistics(self):
        total_waiting_time = [client.service_start_time - client.arrival_time for client in self.served_client]
        mean_waiting_time = sum(total_waiting_time) / len(total_waiting_time) if len(total_waiting_time) > 0 else 0
        
        total_abandonment_time = [client.abandonment_time - client.arrival_time for client in self.abandonment_client]
        total_abandonment = len(self.abandonment_client)
        total_abandonment_type_list = [client.inquiry_type for client in self.abandonment_client]
        total_abandonment_type = [total_abandonment_type_list.count(i) for i in range(len(self.arrival_rate))]
        mean_abandonment_time = sum(total_abandonment_time) / len(total_abandonment_time) if len(total_abandonment_time) > 0 else 0

        total_clients_served_list = [client.inquiry_type for client in self.served_client]
        total_num_clients_served = len(total_clients_served_list)
        total_type_client_served = [total_clients_served_list.count(i) for i in range(len(self.arrival_rate))]
        
        last_event_time = self.time_until if self.time == 0 else self.happened_event[-1][0].time 
        staff_utilization = [[staff.working_time, staff.working_time/last_event_time] for staff in self.staff_pool.staff_list]

        # staff_idle_time = [last_event_time - staff.working_time for staff in self.staff_pool.staff_list]
        staff_idle_time = [staff.idle_time for staff in self.staff_pool.staff_list]

        mean_idle_time = sum(self.idle_time_list) / len(self.idle_time_list) if len(self.idle_time_list) > 0 else 0
        
        result = {}
        result["total_clients"] = self.client_counter
        result["total_served"] = total_num_clients_served
        result["total_abandonment"] = total_abandonment
        result["abandonment_rate"] = total_abandonment/self.client_counter if self.client_counter > 0 else 0
        result["mean_waiting_time"] = mean_waiting_time
        result["mean_abandonment_time"] = mean_abandonment_time
        
        for i,j in enumerate(total_type_client_served):
            result[f"total_served_type_{i}"] = j
        
        for i,j in enumerate(total_abandonment_type):
            result[f"total_abandonment_type_{i}"] = j
        
        for i,j in enumerate(staff_utilization):
            result[f"staff_{i}_working_time"] = j[0]
            result[f"staff_{i}_utilization"] = j[1]
        
        for i,j in enumerate(staff_idle_time):
            result[f"staff_{i}_idle_time"] = j
        
        result["mean_idle_time_action"] = mean_idle_time

        return result
        
    def event_list(self):
        return [{"Time": i[0].time, "ID": i[0].event_id, "Event": str(i[0]), "Staff_1": i[1][0], "Staff_2": i[1][1], "Event_Queue": i[2]} for i in self.happened_event]

    def render(self, mode='human'):
        print(f"Event Queue Length: {self.event_queue.qsize()}")
        print(f"{[[i,str(j)] for i,j in self.event_queue.queue]}\n")

        print(f"Time: {self.time}")
        print(f"Available Staff: {self.staff_pool.num_available_staff()}")
        print(f"Clients Counter: {self.client_counter}")
        print(f"Served Clients: {self.served_clients_counter}")
        print(f"Abandon Clients: {self.abandonment_counter}")
        print(f"Idle Time: {self.idle_time}")
        print(f"Average Waiting Time: {self.avg_waiting_time} \n")
            
        print("--- Staff Status ---")
        for staff in self.staff_pool.staff_list:
            status = 'Available' if staff.available else f"Serving Client {staff.current_client.client_id}: T{staff.current_client.inquiry_type}"
            print(f"Staff {staff.staff_id}: {status}")
            print(f"Staff {staff.staff_id} Client queue: {[f'Client {client.client_id}: T{client.inquiry_type}' for client in staff.staff_queue.queue] if staff.staff_queue.qsize() > 0 else 'empty'}")
            print(f"Number of Clients Served: {staff.num_clients_served}")
            print(f"Number of Abandon Clients: {staff.num_abandon_client}")
            print("\n")
        print("--------------------")



# The Call Centre Environment for Value Iteration
class CallCentreEnvValueIteration(CallCentreEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P_dict = self.build_transition_dict(self.max_staff_queue, self.service_rate, self.arrival_rate, self.abandonment_rate)

    def P(self, state, action):
        """
        Returns a list of (probability, next_state, reward) for a given state and action.
        """
        return self.P_dict[state][action]

    @staticmethod
    def build_transition_dict(max_queue, staff_parameter, arrival_rate, theta_values):
        """
        Builds the deterministic transition dictionary mapping (state, action) 
        to a list with a single (probability=1.0, next_state, reward) tuple.
        """

        def get_next_state(a, b, t, action):
            """Returns a single deterministic next state given the action."""
            next_t = t  # or flip: next_t = 1 - t if desired

            if action == 0:
                new_a = min(a + 1, max_queue)
                new_b = b
            elif action == 1:
                new_a = a
                new_b = min(b + 1, max_queue)
            else:
                raise ValueError("Invalid action")

            return (new_a, new_b, next_t)
            

        def calculate_reward(current_state, action):
            a1, b1, t1 = current_state

            reward = 0
            mean_service_time = {i: sum(j) / len(j) for i, j in enumerate(staff_parameter)}
            queue_len = a1 if action == 0 else b1

            # 1. Waiting time penalty
            expected_wait = queue_len * mean_service_time[action]
            reward -= expected_wait

            # 2. Idle penalty
            if action == 0 and b1 == 0 and a1 > b1:
                reward -= 158
            if action == 1 and a1 == 0 and b1 > a1:
                reward -= 158
            
            # 3. Full queue penalty
            if a1 == max_queue and action == 0:
                reward -= 158
            if b1 == max_queue and action == 1:
                reward -= 158


            return reward

        # All possible states: (senior_queue, junior_queue, task_type)
        states = [(a, b, t) for a in range(max_queue + 1)
                            for b in range(max_queue + 1)
                            for t in (0, 1)]

        P = {}

        for state in states:
            a, b, t = state
            P[state] = {}

            for action in [0, 1]:
                next_state = get_next_state(a, b, t, action)
                # reward = calculate_reward(state, next_state, action, arrival_rate, theta_values)
                reward = calculate_reward(state, action)
                P[state][action] = [(1.0, next_state, reward)]  # deterministic transition

        return P

# The Call Centre Environment for PPO
class CallCentreEnvPPO(CallCentreEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # PPO requires a continuous state space: Normalize state to [0,1]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.staff_pool_size + 1,), dtype=np.float32
        )

    def _get_observation(self):
        obs = np.zeros((self.staff_pool_size +1,), dtype=np.float32)

        for i, staff in enumerate(self.staff_pool.staff_list):
            obs[i] = staff.staff_queue.qsize()

        norm_obs = obs/ self.max_staff_queue

        if not self.event_queue.empty():
            _, event = self.event_queue.queue[0]
            norm_obs[-1] = event.type
        else:
            norm_obs[-1] = 0

        return norm_obs
    