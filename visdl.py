import pandas as pd
import subprocess
from yt_dlp import YoutubeDL

def download_and_trim(movie_url: str, start_time: int, end_time: int, output_path: str):
    # yt-dlpでの動画ダウンロード
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_path,  # ダウンロード先のテンプレート
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([movie_url])

    # ffmpegでのトリミング
    input_file = output_path
    trimmed_file = f"trimmed_{output_path}"
    
    # ffmpegを使って指定した時間範囲でトリミング
    command = [
        'ffmpeg',
        '-i', input_file,
        '-ss', str(start_time),  # 開始時間
        '-to', str(end_time),    # 終了時間
        '-c', 'copy',            # コーデックをコピー
        trimmed_file
    ]
    
    subprocess.run(command, check=True)

    print(f"Trimmed video saved as {trimmed_file}")

def main():
    # CSVファイルのパス
    excel_file_path = "C:/Users/kiyok/PromptingWhisper/visspeech/VisSpeech.csv"
    
    # CSVファイルを読み込む
    df = pd.read_csv(excel_file_path)
    
    # 各列のデータを取得
    yt_id = df['yt_id'][0]
    start_time = df['start_time'][0] / 1000  # ミリ秒を秒に変換
    end_time = df['end_time'][0] / 1000      # ミリ秒を秒に変換
    
    # ダウンロードに使用するURLを作成
    youtube_url = f'https://www.youtube.com/watch?v={yt_id}'
    print(youtube_url)
    # 動画の保存先のテンプレート
    output_path = 'downloaded_video.mp4'
    
    # ダウンロードとトリミング
    download_and_trim(youtube_url, start_time, end_time, output_path)

if __name__ == "__main__":
    main()
