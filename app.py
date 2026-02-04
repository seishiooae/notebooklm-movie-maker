import streamlit as st
import os
import re
import tempfile
import torch
import whisper
import fitz  # PyMuPDF: タイトル抽出用
from pdf2image import convert_from_path
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

# --- 1. PDF分析エンジン ---
def extract_slide_titles(pdf_path):
    titles = {}
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                # 1行目をセクションタイトルとして認識
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                if lines:
                    titles[i + 1] = lines[0]
        doc.close()
    except Exception as e:
        st.error(f"Analysis Error: {e}")
    return titles

# --- UI Layout & Styling ---
st.set_page_config(page_title="AI Video Sync Master", layout="wide", page_icon="⚡")

# タイトルセクション
st.title("⚡ AI Video Sync Master")
st.markdown("#### PDF資料と音声をAIが完全同期。欠落したページを自動検知する進化版。")

# --- クイックスタートガイド ---
with st.expander("🚀 クイックスタート・ガイド", expanded=False):
    st.markdown("""
    ### 🛠️ ワークフロー
    1. **AI Voice Logic**: プロンプトをNotebookLMに入力し、音声を生成。
    2. **Source Upload**: スライドPDFとAudioファイルをロード。
    3. **Core Synthesis**: AIが同期ポイントを解析。見つからないページがあれば警告を表示します。
    """)

# --- System Prompt ---
st.subheader("🔗 System Prompt (Copy & Paste)")
prompt_text = """あなたはプロのプレゼンターとして、スライドの内容を自然な流れで解説してください。
ただし、後で動画編集を行うための目印として、スライドが切り替わるタイミングで、必ず以下のいずれかのフレーズを自然に組み込んでください。

「それでは、1枚目のスライドをご覧ください。」
「続いて、2枚目の内容に移ります。」
「3枚目のページでは、〜について説明しています。」

ルール：
1. 全てのスライドを順番通りに解説すること。
2. スライド番号（数字）と「枚目」または「ページ」という言葉をセットで発言すること。
3. あくまでプレゼンのナレーションとして自然に話し、番号を省略したり飛ばしたりしないでください。"""

st.code(prompt_text, language="text")

st.markdown("---")

# --- アセット・アップロード ---
col1, col2 = st.columns(2)
with col1:
    st.markdown("##### 📁 Slide Assets")
    uploaded_pdf = st.file_uploader("Upload PDF Presentation", type="pdf", label_visibility="collapsed")
with col2:
    st.markdown("##### 🎙️ Audio Assets")
    uploaded_audio = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"], label_visibility="collapsed")

# --- メイン・プロセッシング ---
if uploaded_pdf and uploaded_audio:
    if st.button("🔥 Generate Video Now"):
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_obj.name
        
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("AI Engine is processing..."):
                # PDF処理
                status_text.text("Step 1/4: Analyzing PDF structure...")
                pdf_path = os.path.join(tmpdir, "input.pdf")
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_pdf.read())
                
                slide_titles = extract_slide_titles(pdf_path)
                images = convert_from_path(pdf_path, dpi=120)
                total_slides = len(images)
                image_paths = []
                for i, img in enumerate(images):
                    path = os.path.join(tmpdir, f"slide_{i+1:03d}.png")
                    img.save(path, "PNG")
                    image_paths.append(path)
                progress_bar.progress(25)

                # 音声処理
                status_text.text("Step 2/4: Initializing Audio Engine...")
                audio_ext = os.path.splitext(uploaded_audio.name)[1]
                audio_path = os.path.join(tmpdir, f"input_audio{audio_ext}")
                with open(audio_path, "wb") as f:
                    f.write(uploaded_audio.read())
                progress_bar.progress(50)

                # Whisper分析
                status_text.text("Step 3/4: Transcribing and Syncing...")
                model = whisper.load_model("base", device="cpu")
                result = model.transcribe(audio_path, language="ja", fp16=False)

                markers = [{"slide": 1, "start": 0.0}]
                found_slides = {1}

                # キーワード検索
                for segment in result['segments']:
                    text = segment['text']
                    match = re.search(r"(\d+)\s*(枚目|ページ|スライド)", text)
                    if match:
                        num = int(match.group(1))
                        if num not in found_slides and num <= total_slides:
                            markers.append({"slide": num, "start": segment['start']})
                            found_slides.add(num)

                # タイトル補完
                for page_num, title in slide_titles.items():
                    if page_num not in found_slides and len(title) > 3:
                        for segment in result['segments']:
                            if title in segment['text']:
                                markers.append({"slide": page_num, "start": segment['start']})
                                found_slides.add(page_num)
                                break
                
                # --- 追加機能: 欠番のチェックと表示 ---
                missing_slides = [i for i in range(1, total_slides + 1) if i not in found_slides]
                if missing_slides:
                    st.warning(f"⚠️ 以下のスライドが見つからず、スキップされました: {missing_slides}")
                    st.info("解決策: NotebookLMのプロンプトで、これらのスライド番号をハッキリ言うように指示するか、スライドの1行目（タイトル）を音声に含めてください。")
                else:
                    st.success("✨ すべてのスライドの同期ポイントを特定しました！")

                progress_bar.progress(75)

                # ビデオ・レンダリング
                status_text.text("Step 4/4: Final Rendering...")
                markers = sorted(markers, key=lambda x: x["start"])
                audio_clip = AudioFileClip(audio_path)
                clips = []
                
                for i in range(len(markers)):
                    idx = markers[i]["slide"] - 1
                    if idx < len(image_paths):
                        start_time = markers[i]["start"]
                        end_time = markers[i+1]["start"] if i+1 < len(markers) else audio_clip.duration + 1.0
                        duration = end_time - start_time
                        if duration > 0:
                            clip = ImageClip(image_paths[idx]).with_duration(duration)
                            clips.append(clip)

                if clips:
                    final_video = concatenate_videoclips(clips).with_audio(audio_clip)
                    output_file = os.path.join(tmpdir, "final_video.mp4")
                    final_video.write_videofile(output_file, fps=5, codec="libx264", audio_codec="aac")
                    
                    progress_bar.progress(100)
                    status_text.text("Synthesis Complete.")
                    
                    with open(output_file, "rb") as f:
                        st.download_button(label="📥 Download Exported Video", data=f, file_name="ai_sync_video.mp4", mime="video/mp4")
                    
                    final_video.close()
                    audio_clip.close()
                else:
                    st.error("Fatal Error: 有効なスライドマーカーが1つも見つかりませんでした。")

        except Exception as e:
            st.error(f"System Error: {e}")
        finally:
            tmpdir_obj.cleanup()
