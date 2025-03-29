from flask import Flask, request, jsonify, render_template
import openai
import os
from datetime import datetime
import json
app = Flask(__name__)
# 设置 OpenAI API 密钥
#从api_key.json中读取密钥


# 日志文件路径
LOG_FILE = "log/voice2text_log.json"

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

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)