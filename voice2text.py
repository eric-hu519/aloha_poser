from flask import Flask, request, jsonify, render_template
import openai
import os
from datetime import datetime
import json
import multiprocessing
import time
import traceback
app = Flask(__name__)
# 设置 OpenAI API 密钥
#从api_key.json中读取密钥


# 日志文件路径
LOG_FILE = "log/voice2text_log.json"
#setting openai api key
with open('api_key.json', 'r') as f:
    api_key = json.load(f).get('api_key')
openai.api_key = api_key

# 全局变量存储进程状态
execution_processes = {}
execution_status = {}
# 主页面，提供录音功能
@app.route('/')
def index():
    return render_template('record.html')

# 处理音频上传并转换为文本
@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # 保存音频文件
    audio_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(audio_path)
    #成功保存提示
    #return jsonify({'message': 'Audio file uploaded successfully!'})
    # 调用 OpenAI Whisper API 进行语音转文字
    try:
        with open(audio_path, 'rb') as audio:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                language="zh",  # 语言：中文
                #response_format="text"  # 返回文本

            )
        
        # 提取中文文本
        original_text = response.text 
        if not original_text:
            return jsonify({'error': 'No text found in the audio'}), 400
        # 调用 OpenAI GPT 进行中文到英文翻译
        translation_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Translate the following Chinese text to English:"},
                {"role": "user", "content": original_text}
            ]
        )
        
        # 提取翻译结果
        translated_text = translation_response.choices[0].message.content

        # 返回中英文结果
        return jsonify({
            'original_text': original_text,
            'translated_text': translated_text
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 保存翻译结果到 log.json
@app.route('/save', methods=['POST'])
def save_to_log():
    try:
        data = request.json
        original_text = data.get('original_text')
        translated_text = data.get('translated_text')

        if not original_text or not translated_text:
            return jsonify({'error': 'Invalid data'}), 400

        # 记录当前时间
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 构造日志条目
        log_entry = {
            "time": timestamp,
            "zh": original_text,
            "en": translated_text
        }

        # 如果日志文件不存在，创建一个空的 JSON 文件
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
                json.dump([], log_file, ensure_ascii=False, indent=4)

        # 读取现有日志文件并追加新条目
        with open(LOG_FILE, 'r', encoding='utf-8') as log_file:
            logs = json.load(log_file)

        logs.append(log_entry)

        # 将更新后的日志写回文件
        with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
            json.dump(logs, log_file, ensure_ascii=False, indent=4)

        return jsonify({'message': 'Translation saved successfully!'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# 添加一个新的 API 处理用户请求并生成任务
@app.route('/process_query', methods=['POST'])
def process_query():
    try:
        data = request.json
        user_query = data.get('user_query')
        use_cached = data.get('use_cached', False)

        if not user_query:
            return jsonify({'error': 'Invalid user query'}), 400

        # 执行 Aloha Poser 的任务分解逻辑
        from aloha_poser import lmp_call

        result = lmp_call(user_query=user_query, use_cached=use_cached)

        # 返回生成的任务和动作
        return jsonify({'success': True, 'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 执行机器人动作的函数（在独立进程中运行）
def execute_robot_actions(action_sequence, process_id):
    try:
        # 导入并初始化机器人控制器
        from robot_controller_v2 import Robot_Controller
        
        # 初始化控制器
        controller = Robot_Controller(test_camera=False,api_call=True)
        
        # 执行动作序列
        controller.run(action_sequence)
        
        # 更新状态为成功
        execution_status[process_id] = {
            'status': 'completed',
            'message': 'Action sequence executed successfully',
            'error': None
        }
        
    except Exception as e:
        # 更新状态为失败
        execution_status[process_id] = {
            'status': 'failed',
            'message': f'Execution failed: {str(e)}',
            'error': traceback.format_exc()
        }

# 添加执行动作序列的API
@app.route('/execute_actions', methods=['POST'])
def execute_actions():
    try:
        data = request.json
        action_sequence = data.get('action_sequence')
        
        if not action_sequence:
            return jsonify({'error': 'No action sequence provided'}), 400
        
        # 生成唯一的进程ID
        process_id = f"exec_{int(time.time() * 1000)}"
        
        # 初始化状态
        execution_status[process_id] = {
            'status': 'running',
            'message': 'Executing action sequence...',
            'error': None
        }
        
        # 创建并启动新进程
        process = multiprocessing.Process(
            target=execute_robot_actions,
            args=(action_sequence, process_id)
        )
        process.start()
        
        # 存储进程引用
        execution_processes[process_id] = process
        
        return jsonify({
            'success': True, 
            'process_id': process_id,
            'message': 'Execution started in background process'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start execution: {str(e)}'}), 500

# 查询执行状态的API
@app.route('/execution_status/<process_id>', methods=['GET'])
def get_execution_status(process_id):
    try:
        if process_id not in execution_processes:
            return jsonify({'error': 'Process not found'}), 404
            
        process = execution_processes[process_id]
        
        # 检查进程是否还在运行
        if process.is_alive():
            status = execution_status.get(process_id, {
                'status': 'running',
                'message': 'Executing action sequence...',
                'error': None
            })
        else:
            # 进程已结束，获取最终状态
            status = execution_status.get(process_id, {
                'status': 'completed' if process.exitcode == 0 else 'failed',
                'message': 'Process completed' if process.exitcode == 0 else 'Process failed',
                'error': None if process.exitcode == 0 else f'Exit code: {process.exitcode}'
            })
            
            # 清理已完成的进程
            if process_id in execution_processes:
                execution_processes[process_id].join()  # 确保进程完全结束
                del execution_processes[process_id]
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 停止执行的API
@app.route('/stop_execution/<process_id>', methods=['POST'])
def stop_execution(process_id):
    try:
        if process_id not in execution_processes:
            return jsonify({'error': 'Process not found'}), 404
            
        process = execution_processes[process_id]
        
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)  # 等待5秒
            
            if process.is_alive():
                process.kill()  # 强制终止
                
            execution_status[process_id] = {
                'status': 'stopped',
                'message': 'Execution stopped by user',
                'error': None
            }
        
        # 清理进程
        if process_id in execution_processes:
            del execution_processes[process_id]
            
        return jsonify({'success': True, 'message': 'Execution stopped'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 设置多进程启动方法
    multiprocessing.set_start_method('spawn', force=True)
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)