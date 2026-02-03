import streamlit as st
import os
import re
import tempfile
import torch
import whisper
import fitz  # PyMuPDF: タイトル抽出用
from pdf2image import convert_from_path
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

# --- 1. PDFから各ページの1行目（タイトル）を抽出する関数 ---
def extract_slide_titles(pdf_path):
    titles = {}
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                # 空行を除いた最初の行をタイトルとする
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                if lines:
                    titles[i + 1] = lines[0]
        doc.close()
    except Exception as e:
        st.error(f"テキスト抽出エラー: {e}")
    return titles

st.set_page_config(page_title="NotebookLM Video Maker", layout="wide")
st.title("🎬 NotebookLM 自動動画合成ツール")
st.info("PDFのタイトルと音声の文脈をAIが理解して、自動で動画を組み立てます。")

# ファイルアップローダー
col1, col2 = st.columns(2)
with col1:
    uploaded_pdf = st.file_uploader("スライドPDFをアップロード", type="pdf")
with col2:
    uploaded_audio = st.file_uploader("音声ファイルをアップロード", type=["wav", "mp3", "m4a"])

if uploaded_pdf and uploaded_audio:
    if st.button("🚀 動画生成を開始"):
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_obj.name
        
        try:
            with st.spinner("AIが解析と合成を行っています。数分かかる場合があります..."):
                # 1. PDFの保存とデータ抽出
                pdf_path = os.path.join(tmpdir, "input.pdf")
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_pdf.read())
                
                # タイトル抽出
                slide_titles = extract_slide_titles(pdf_path)
                
                # 画像変換
                images = convert_from_path(pdf_path, dpi=120)
                image_paths = []
                for i, img in enumerate(images):
                    path = os.path.join(tmpdir, f"slide_{i+1:03d}.png")
                    img.save(path, "PNG")
                    image_paths.append(path)

                # 2. 音声の保存
                audio_ext = os.path.splitext(uploaded_audio.name)[1]
                audio_path = os.path.join(tmpdir, f"input_audio{audio_ext}")
                with open(audio_path, "wb") as f:
                    f.write(uploaded_audio.read())

                # 3. Whisperによる音声解析 (CPUを使用)
                st.write("🔍 音声の内容を分析中...")
                model = whisper.load_model("base", device="cpu")
                result = model.transcribe(audio_path, language="ja", fp16=False)

                # マーカー特定（スライド番号 ＋ タイトル補完）
                markers = [{"slide": 1, "start": 0.0}]
                found_slides = {1}

                # キーワード検索
                for segment in result['segments']:
                    text = segment['text']
                    match = re.search(r"(\d+)\s*(枚目|ページ|スライド)", text)
                    if match:
                        num = int(match.group(1))
                        if num not in found_slides and num <= len(image_paths):
                            markers.append({"slide": num, "start": segment['start']})
                            found_slides.add(num)

                # タイトルによる補完
                for page_num, title in slide_titles.items():
                    if page_num not in found_slides and len(title) > 3:
                        for segment in result['segments']:
                            if title in segment['text']:
                                markers.append({"slide": page_num, "start": segment['start']})
                                found_slides.add(page_num)
                                st.write(f"✨ タイトル一致で補完: '{title}' (Slide {page_num})")
                                break

                # 時間順にソート
                markers = sorted(markers, key=lambda x: x["start"])

                # 4. 動画の組み立て
                st.write("🎞️ 動画をレンダリング中...")
                audio_clip = AudioFileClip(audio_path)
                clips = []
                
                for i in range(len(markers)):
                    idx = markers[i]["slide"] - 1
                    if idx < len(image_paths):
                        start_time = markers[i]["start"]
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
                    
                    # サーバー用に libx264 を使用
                    final_video.write_videofile(
                        output_file, 
                        fps=5, 
                        codec="libx264", 
                        audio_codec="aac"
                    )
                    
                    st.success("✅ 動画が完成しました！")
                    with open(output_file, "rb") as f:
                        st.download_button(
                            label="動画をダウンロード",
                            data=f,
                            file_name="notebooklm_video.mp4",
                            mime="video/mp4"
                        )
                    
                    final_video.close()
                    audio_clip.close()
                else:
                    st.error("スライドの切り替えポイントを特定できませんでした。")

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
        finally:
            tmpdir_obj.cleanup()