import openai
import os

# 设置 OpenAI API 密钥
# 你可以直接在代码中设置，也可以通过读取环境变量或配置文件获取
#openai.api_key = 

def transcribe_audio(file_path):
    try:
        with open(file_path, 'rb') as audio:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                language="zh",  # 语言：中文
                #response_format="text"  # 返回文本
                
            )
        return response.text
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def main():
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        print(f"文件夹 '{uploads_dir}' 不存在，请确认上传目录。")
        return

    # 遍历uploads目录下所有wav文件
    for file in os.listdir(uploads_dir):
        if file.lower().endswith(".wav"):
            file_path = os.path.join(uploads_dir, file)
            print(f"正在处理文件: {file_path}")
            transcription = transcribe_audio(file_path)
            if transcription is not None:
                print("转写结果：")
                print(transcription)
            else:
                print("转写失败。")
            print("-" * 40)

if __name__ == "__main__":
    main()