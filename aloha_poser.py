import array
from operator import is_
import os
from pydantic import BaseModel
from openai import OpenAI
from typing import List, Union
import json
import jsonschema
#import hashlib
from sympy import N
from robot_controller import robot_controller
with open('api_key.json', 'r') as file:
    API_KEY = json.load(file)['api_key']

TASK_PLANNER = 'prompts/task_planner_prompt.txt'
ACTION_EXAMPLE = 'prompts/action_example_prompt.txt'
USER_QUERY = 'grasp the red straw and lift it above the cup, move conterclock-wise above the cup and then drop the straw.'
SYSTEM_PROMPT =  "You are a helpful assistant that pays attention to the user's instructions and writes good code in required format for operating a robot arm in a tabletop environment."
ASSISTANT_PROMPT = 'Got it. I will complete what you give me next.'
CONTEXT_PROMPT = "I would like you to help me write the formated code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation. Do not add any comment. I will first give you the context of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up."
OBJ = 'objects = [\'red straw\', \'cup\']'


class TaskReasoning(BaseModel):
    steps: List[str]


class LMP:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def api_call(self, prompt:str, query:str, model:str, response_format=Union[TaskReasoning,dict], obj_context: str = None):
        prompt = f"I would like you to help me write formated code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation (comment in code is okay). I will first give you the context of the code below:\n\n```\n{prompt}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up."
        query = '#Query:' + query
        if obj_context is not None:
            query = obj_context.strip() + '\n' + query
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
    
    def load_prompt(self, file_path: str):
        with open(file_path, 'r') as file:
            prompts = file.read().strip()
        return prompts

def main():
    lmp = LMP(api_key=API_KEY)
    prompt = lmp.load_prompt(TASK_PLANNER)
    print(f'user query: {USER_QUERY}')
    is_cached = False
    #use cached response
    if os.path.exists('task_reasoning_test.json'):
        with open('task_reasoning_test.json', 'r') as file:
            response = json.load(file)
            if response.get('query') == USER_QUERY and response.get('steps') is not None:
                print('Using cached response')
                is_cached = True
                #response = TaskReasoning(**response)
    if not is_cached:
        response = lmp.api_call(prompt=prompt, 
                                    query=USER_QUERY, 
                                    model='gpt-4o-2024-08-06', 
                                    response_format=TaskReasoning,
                                    obj_context=OBJ)

    #save the task reasoning
        with open('task_reasoning_test.json', 'w') as file:
            #add query to the response
            response = response.dict()
            response['query'] = USER_QUERY
            json.dump(response, file)
            
    #load json schema as prompt
    with open("ability_api_schema.json", 'r') as file:
        schema = json.load(file)
    action_prompt = lmp.load_prompt(ACTION_EXAMPLE)
    action_sequence = []
    print("Steps:")
    for step in response['steps']:
        print(step)
        act_response= lmp.api_call(prompt=action_prompt, 
                              query=step, 
                              model='gpt-4o-2024-08-06', 
                              response_format=schema)
                              
        #parse the response
        act_response = json.loads(act_response)
        for action in act_response['actions']:
            action_sequence.append(action)
    #save the action sequence
    with open('action_sequence_test.json', 'w') as file:
        json.dump(action_sequence, file)
    # #check json schema
    # with open('action_check_schema.json', 'r') as file:
    #     schema = json.load(file)
    # try:
    #     jsonschema.validate(action_sequence, schema)
    #     print('Action sequence is valid')
    # except jsonschema.exceptions.ValidationError as e:
    #     print('Action sequence is invalid')
    #     print(e)
    print('Action sequence:\n')
    for action in action_sequence:
        print(action)
    #press y to run action or abort
    run = input('Run action sequence? (y/n): ')
    if run == 'y':
        controller = robot_controller()
        controller.run(action_sequence)
    else:
        print('Aborted')


if __name__ == '__main__':
    main()