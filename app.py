import streamlit as st
import os
import re
import tempfile
import torch
import whisper
import fitz  # PyMuPDF: タイトル抽出用
from pdf2image import convert_from_path
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

# --- 1. PDFからタイトルを抽出する関数 ---
def extract_slide_titles(pdf_path):
    titles = {}
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                # 最初の行をタイトルとして抽出
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                if lines:
                    titles[i + 1] = lines[0]
        doc.close()
    except Exception as e:
        st.error(f"PDFテキスト解析エラー: {e}")
    return titles

# --- ページ設定 ---
st.set_page_config(page_title="NotebookLM Video Maker", layout="wide", page_icon="🎬")

# --- タイトル表示 ---
st.title("🎬 NotebookLM 自動動画合成ツール")
st.markdown("PDF資料と音声をAIが解析し、1つの動画に自動でまとめます。")

# --- 使い方の手順（折りたたみ式） ---
with st.expander("📖 使い方の手順をチェックする", expanded=False):
    st.markdown("""
    ### 🎬 動画作成の3ステップ
    
    1.  **NotebookLMで音声を生成**
        - 下記の「専用プロンプト」をコピーして、NotebookLMの「音声のカスタマイズ」欄に貼り付けてください。
        - 生成された音声（.m4aや.wav）をダウンロードします。
    2.  **ファイルをアップロード**
        - このサイトに「元のスライドPDF」と「ダウンロードした音声」をセットします。
    3.  **動画を生成してダウンロード**
        - 「動画生成を開始」ボタンを押すと、AIが音声を解析し、スライドを自動で切り替えます。
        - 完成したらダウンロードボタンが表示されます。
    """)

# --- 専用プロンプト表示（コピーボタン付き） ---
st.subheader("📋 NotebookLM 貼り付け用プロンプト")
st.info("音声を生成する際、以下のプロンプトを使用すると、AIがスライドの切り替えタイミングを正確に教えてくれるようになります。")

prompt_text = """あなたはプロのプレゼンターとして、スライドの内容を自然な流れで解説してください。
ただし、後で動画編集を行うための目印として、スライドが切り替わるタイミングで、必ず以下のいずれかのフレーズを自然に組み込んでください。

「それでは、1枚目のスライドをご覧ください。」
「続いて、2枚目の内容に移ります。」
「3枚目のページでは、〜について説明しています。」

ルール：
1. 全てのスライドを順番通りに解説すること。
2. スライド番号（数字）と「枚目」または「ページ」という言葉をセットで発言すること。
3. あくまでプレゼンのナレーションとして自然に話し、番号を省略したり飛ばしたりしないでください。"""

# st.codeを使うと、画面上で1クリックコピーが可能になります
st.code(prompt_text, language="text")

st.markdown("---")

# --- アップロードセクション ---
col1, col2 = st.columns(2)
with col1:
    uploaded_pdf = st.file_uploader("1. スライドPDFを選択してください", type="pdf")
with col2:
    uploaded_audio = st.file_uploader("2. 音声ファイルを選択してください", type=["wav", "mp3", "m4a"])

# --- 処理メイン ---
if uploaded_pdf and uploaded_audio:
    if st.button("🚀 動画生成を開始する"):
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_obj.name
        
        try:
            with st.spinner("AIが資料と音声を解析しています... しばらくお待ちください。"):
                # PDFの保存
                pdf_path = os.path.join(tmpdir, "input.pdf")
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_pdf.read())
                
                # タイトル抽出
                slide_titles = extract_slide_titles(pdf_path)
                
                # PDFを画像に変換
                images = convert_from_path(pdf_path, dpi=120)
                image_paths = []
                for i, img in enumerate(images):
                    path = os.path.join(tmpdir, f"slide_{i+1:03d}.png")
                    img.save(path, "PNG")
                    image_paths.append(path)

                # 音声の保存
                audio_ext = os.path.splitext(uploaded_audio.name)[1]
                audio_path = os.path.join(tmpdir, f"input_audio{audio_ext}")
                with open(audio_path, "wb") as f:
                    f.write(uploaded_audio.read())

                # Whisper音声解析 (CPU環境用に最適化)
                st.write("🔍 音声の中からスライドの切り替えポイントを探しています...")
                model = whisper.load_model("base", device="cpu")
                result = model.transcribe(audio_path, language="ja", fp16=False)

                # 同期ポイントの特定
                markers = [{"slide": 1, "start": 0.0}]
                found_slides = {1}

                # 手順A: キーワード（〇枚目など）で検索
                for segment in result['segments']:
                    text = segment['text']
                    match = re.search(r"(\d+)\s*(枚目|ページ|スライド)", text)
                    if match:
                        num = int(match.group(1))
                        if num not in found_slides and num <= len(image_paths):
                            markers.append({"slide": num, "start": segment['start']})
                            found_slides.add(num)

                # 手順B: タイトル名でスマート補完
                for page_num, title in slide_titles.items():
                    if page_num not in found_slides and len(title) > 3:
                        for segment in result['segments']:
                            if title in segment['text']:
                                markers.append({"slide": page_num, "start": segment['start']})
                                found_slides.add(page_num)
                                st.write(f"✨ タイトル一致で補完: '{title}' (Slide {page_num})")
                                break

                # 時間順に整理
                markers = sorted(markers, key=lambda x: x["start"])

                # 動画の合成
                st.write("🎞️ 動画を組み立て中...")
                audio_clip = AudioFileClip(audio_path)
                clips = []
                
                for i in range(len(markers)):
                    idx = markers[i]["slide"] - 1
                    if idx < len(image_paths):
                        start_time = markers[i]["start"]
                        # 次のページまで
                        if i + 1 < len(markers):
                            end_time = markers[i+1]["start"]
                        else:
                            end_time = audio_clip.duration + 1.0
                        
                        duration = end_time - start_time
                        if duration > 0:
                            clip = ImageClip(image_paths[idx]).with_duration(duration)
                            clips.append(clip)

                if clips:
                    final_video = concatenate_videoclips(clips).with_audio(audio_clip)
                    output_file = os.path.join(tmpdir, "final_video.mp4")
                    
                    # サーバー環境用にCPUエンコーダー(libx264)を指定
                    final_video.write_videofile(output_file, fps=5, codec="libx264", audio_codec="aac")
                    
                    st.success("✅ 動画が完成しました！下のボタンから保存してください。")
                    with open(output_file, "rb") as f:
                        st.download_button(
                            label="📥 完成した動画をダウンロード",
                            data=f,
                            file_name="ai_generated_presentation.mp4",
                            mime="video/mp4"
                        )
                    
                    final_video.close()
                    audio_clip.close()
                else:
                    st.error("スライドの切り替えポイントが見つかりませんでした。プロンプトを見直してみてください。")

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
        finally:
            tmpdir_obj.cleanup()
