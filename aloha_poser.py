import array
from operator import is_
import os
import re
from click import prompt
from pydantic import BaseModel
from openai import OpenAI
from typing import List, Union
import json
import jsonschema
import uuid
#import hashlib
#from robot_controller import robot_controller
with open('api_key.json', 'r') as file:
    API_KEY = json.load(file)['api_key']

TASK_PLANNER = 'prompts/task_planner_prompt.txt'
ACTION_EXAMPLE = 'prompts/action_example_prompt.txt'
USER_QUERY = 'Put the blue block into the cup and then hand me over the cup.'
SYSTEM_PROMPT =  "You are a helpful assistant that pays attention to the user's instructions and writes good code in required format for operating a robot arm in a tabletop environment."
ASSISTANT_PROMPT = 'Got it. I will complete what you give me next.'
CONTEXT_PROMPT = "I would like you to help me write the formated code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation. Do not add any comment. I will first give you the context of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up."
MODEL = 'gpt-4o'

#TODO: parse部分需要重写
class TaskReasoning(BaseModel):
    steps: List[str]


class LMP:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.logger = Logging_Handler()

    def api_call(self, prompt:str, query:str, model:str, response_format=Union[TaskReasoning,dict]):
        prompt = f"I would like you to help me write formated code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation (comment in code is okay). I will first give you the context of the code below:\n\n```\n{prompt}\n```\n\nNote that x is back to front(the front is positive), y is right to left(left is positive), and z is bottom to up(up is positive). If the query doesn't ask for detect action, then you don't have to include the corresponding detect action in your code."
        query = '#Query: ' + query
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ASSISTANT_PROMPT},
            {"role": "user", "content": query},
        ]
        if response_format is not None:
            if not isinstance(response_format, dict):
                response_format = TaskReasoning
            else:
                response_format = {"type": "json_schema", "json_schema": response_format}
        try:
            completion = self.client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format
            )
        except Exception as e:
            print(e)
            return None
        result = completion.choices[0].message.parsed
        if result is not None:
            return result
        else:
            return completion.choices[0].message.content
    
    def load_planner_prompt(self):
        prompt_file = self.logger.get_all_quest_hist()
        prompt = ""
        for item in prompt_file:
            quest_content = item.get('quest_content',"")
            steps = item.get('steps', [])
            prompt += f"#Query: {quest_content}\n"
            prompt += "[\n"
            for step in steps:
                prompt += f' "{step}",\n'
            prompt = prompt.rstrip(",\n") + "\n"
            prompt +=  "]\n# done\n\n"
        return prompt

    def load_action_prompt(self):
        #load the action prompt
        prompt_file = self.logger.get_all_action_hist()
        prompt = ""
        for item in prompt_file:
            quest_content = item.get('quest_content',"")
            actions = item.get('actions', [])
            prompt += f"#Query: {quest_content}\n"
            prompt += "[\n"
            for action in actions:
                prompt += f' "{action}",\n'
            prompt = prompt.rstrip(",\n") + "\n"
            prompt +=  "]\n# done\n\n"
        return prompt

class Logging_Handler:
    def __init__(self):
        self.quest_log_file_path = "log/user_quest_log.json"
        self.action_log_file_path = "log/action_sequence_log.json"
        if not os.path.exists(self.quest_log_file_path):
            with open(self.quest_log_file_path, 'w') as file:
                json.dump([], file)
        if not os.path.exists(self.action_log_file_path):
            with open(self.action_log_file_path, 'w') as file:
                json.dump([], file)
        self.temp_action = []
    
    def log_checker(self, id: str)-> bool:
        with open(self.quest_log_file_path, 'r') as file:
            logs = json.load(file)
            for log in logs:
                if log.get('quest_ID') == id:
                    return True
        return False
    
    def log_task_reasoning(self, user_quest:str, task_reasoning:dict):
        with open(self.quest_log_file_path, 'r') as file:
            logs = json.load(file)
        #check if the quest is already logged
        if self.check_user_quest(user_quest) is not None:
            print(f"Quest '{user_quest}' already logged, skipping...")
            return
        else:
            #get the latest quest ID
            if len(logs) > 0:
                quest_id = str(int(logs[-1]['quest_ID']) + 1)
            else:
                quest_id = '0'
            logs.append({
                'quest_ID': quest_id,
                'quest_content': user_quest,
                'steps': task_reasoning.get('steps', []),
            })
            with open(self.quest_log_file_path, 'w') as file:
                json.dump(logs, file, indent=4)
                print(f"Task reasoning for quest '{user_quest}' logged with ID: {quest_id}")
        return quest_id
    
    #save the action sequence to temp local variable
    def temp_action_logger(self, step_id: str, step_quest:str, action: list):
        temp_action = {
            'step_ID': step_id,
            'quest_content': step_quest,
            'action': action
        }
        self.temp_action.append(temp_action)
        #print(f"Temporary action logged for step ID '{step_id}'.")

    def log_action_sequence(self, quest_id: str, action_sequence: List[dict]):
        with open(self.action_log_file_path, 'r') as file:
            logs = json.load(file)
        if len(self.temp_action) > 0:
            for action in self.temp_action:
                    action['quest_ID'] = quest_id
                    logs.append(action)
            with open(self.action_log_file_path, 'w') as file:
                json.dump(logs, file, indent=4)
                print(f"Action sequence for quest ID '{quest_id}' logged.")
            self.temp_action = []  # Clear the temporary actions after logging
        else:
            raise ValueError(f"No temporary actions found for quest ID '{quest_id}'. Please log actions before saving the sequence.")
    def clear_temp_logs(self):
        self.temp_action = []
        print("Logs cleared.")

    def get_quest_hist(self, quest_id: str) -> List[dict]:
        with open(self.quest_log_file_path, 'r') as file:
            logs = json.load(file)
            for log in logs:
                if log.get('quest_ID') == quest_id:
                    #return in json format
                    return {
                        'quest_content': log.get('quest_content'),
                        'steps': log.get('steps', [])
                    }
        return {}
    
    def get_action_hist(self, quest_id: str) -> List[dict]:
        with open(self.action_log_file_path, 'r') as file:
            logs = json.load(file)
            step_ID = 0
            actions = []
            for log in logs:
                if log.get('quest_ID') == quest_id:
                    if log.get('step_ID') == str(step_ID):
                        actions.append(
                            {
                                "quest_content": log.get('quest_content'),
                                "actions": log.get('action')
                            }
                        )
                        step_ID += 1
        return actions
    
    
    def get_all_quest_hist(self) -> List[dict]:
        with open(self.quest_log_file_path, 'r') as file:
            logs = json.load(file)
            return logs
    def get_all_action_hist(self) -> List[dict]:
        with open(self.action_log_file_path, 'r') as file:
            logs = json.load(file)
            return logs
    #check if there is a same quest in the log
    def check_user_quest(self, quest:str):
        with open(self.quest_log_file_path, 'r') as file:
            logs = json.load(file)
            for log in logs:
                if log.get('quest_content') == quest:
                    return log.get('quest_ID')
        return None
    #check if there is a same action in the log
    def check_action_sequence(self, action_sequence: List[dict]) -> bool:
        with open(self.action_log_file_path, 'r') as file:
            logs = json.load(file)
            for log in logs:
                if log.get('action_sequence') == action_sequence:
                    return True
        return False


def main(use_cached: bool = False):
    lmp = LMP(api_key=API_KEY)
    #prompt = lmp.load_prompt(TASK_PLANNER)
    logger = Logging_Handler()
    quest_prompt = lmp.load_planner_prompt()
    print(f'user query: {USER_QUERY}')
    if use_cached:
        #check query if exist
        quest_id = logger.check_user_quest(USER_QUERY)
        if quest_id is not None:
            #load the quest from log
            response = logger.get_quest_hist(quest_id)
            print('Quest already logged, loading from log...')
        else:
            response = lmp.api_call(prompt=quest_prompt, 
                                        query=USER_QUERY, 
                                        model=MODEL, 
                                        response_format=TaskReasoning,
                                        )
            response = response.dict()
    else:
            response = lmp.api_call(prompt=quest_prompt, 
                                        query=USER_QUERY, 
                                        model=MODEL, 
                                        response_format=TaskReasoning,
                                        )
            response = response.dict()
    #load json schema as prompt
    with open("ability_api_schema.json", 'r') as file:
        schema = json.load(file)
    action_prompt = lmp.load_action_prompt()
    action_sequence = []
    logger.clear_temp_logs()
    print("Steps:")
    step_id = 0
    for step in response['steps']:
        print(step)
        if not use_cached:
            act_response= lmp.api_call(prompt=action_prompt, 
                                query=step, 
                                model=MODEL, 
                                response_format=schema)
                                
            #temp log actions
            act_response = json.loads(act_response)
            logger.temp_action_logger(str(step_id), step, act_response['actions'])
        else:
            #check if the action is already logged
            if quest_id is not None:
                #load the action from log
                act_response = logger.get_action_hist(quest_id)
            else:
                act_response = lmp.api_call(prompt=action_prompt, 
                                query=step, 
                                model=MODEL, 
                                response_format=schema)
                act_response = json.loads(act_response)
                logger.temp_action_logger(str(step_id), step, act_response['actions'])
        step_id += 1
        print(act_response['actions'])
        for action in act_response['actions']:
            action_sequence.append(action)
    print('Action sequence:\n')
    for action in action_sequence:
        print(action)
    #press y to run action or abort
    run = input('Save current quest? (y/n): ')
    if run == 'y':
        quest_id = logger.log_task_reasoning(USER_QUERY, response)
        if quest_id is not None:
            logger.log_action_sequence(quest_id, action_sequence)
            print(f"Quest '{USER_QUERY}' logged with ID: {quest_id}")
    else:
        logger.clear_temp_logs()
        print('Aborted')

from concurrent.futures import ThreadPoolExecutor, as_completed

def lmp_call(user_query: str, use_cached: bool = False):
    lmp = LMP(api_key=API_KEY)
    logger = Logging_Handler()

    quest_prompt = lmp.load_planner_prompt()

    # 如果使用缓存，则加载缓存数据
    if use_cached:
        quest_id = logger.check_user_quest(user_query)
        if quest_id is not None:
            response = logger.get_quest_hist(quest_id)
        else:
            response = lmp.api_call(prompt=quest_prompt, query=user_query, model=MODEL, response_format=TaskReasoning)
            response = response.dict()
    else:
        response = lmp.api_call(prompt=quest_prompt, query=user_query, model=MODEL, response_format=TaskReasoning)
        response = response.dict()

    # 加载 JSON schema
    with open("ability_api_schema.json", 'r') as f:
        schema = json.load(f)

    # 加载 action prompt
    action_prompt = lmp.load_action_prompt()

    step_results = []

    # 定义并发处理函数
    def fetch_actions(index, step_text):
        act_response = lmp.api_call(
            prompt=action_prompt,
            query=step_text,
            model=MODEL,
            response_format=schema
        )
        return index, step_text, json.loads(act_response)['actions']

    # 使用线程池并发获取每一步的 action
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(fetch_actions, idx, step)
            for idx, step in enumerate(response['steps'])
        ]
        for future in as_completed(futures):
            idx, step_text, actions = future.result()
            step_results.append((idx, {"step": step_text, "actions": actions}))

    # 排序并生成最终有序结果
    step_results.sort(key=lambda x: x[0])  # 按 index 排序
    ordered_results = [item[1] for item in step_results]

    return ordered_results
if __name__ == '__main__':
    main()
    #test_logger()
    #test_planner()