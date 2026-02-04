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
st.set_page_config(page_title="NotebookLM Video Maker", layout="wide", page_icon="🎬")

# タイトルセクション（初期の文言をテックレイアウトで）
st.title("🎬 NotebookLM 自動動画合成ツール")
st.markdown("#### PDFのタイトルと音声の文脈をAIが理解して、自動で動画を組み立てます。")

# --- クイックスタートガイド ---
with st.expander("🚀 使い方の手順をチェックする", expanded=False):
    st.markdown("""
    ### 🛠️ ワークフロー
    
    1.  **NotebookLMで音声を準備**: 
        - 下記の「専用プロンプト」をコピーしてNotebookLMに入力し、音声を生成。
    2.  **ファイルをアップロード**:
        - このサイトに「スライドPDF」と「音声ファイル」をセットします。
    3.  **動画生成**:
        - AIが音声を解析し、スライドを自動で切り替えます。見つからないページがあれば警告を表示します。
    """)

# --- System Prompt for NotebookLM ---
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

st.code(prompt_text, language="text")

st.markdown("---")

# --- アップロードセクション ---
col1, col2 = st.columns(2)
with col1:
    st.markdown("##### 📁 スライドPDFをアップロード")
    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
with col2:
    st.markdown("##### 🎙️ 音声ファイルをアップロード")
    uploaded_audio = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"], label_visibility="collapsed")

# --- メイン・プロセッシング ---
if uploaded_pdf and uploaded_audio:
    if st.button("🔥 動画生成を開始する"):
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_obj.name
        
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("AIが資料と音声を解析しています..."):
                # 1. PDF処理
                status_text.text("Step 1/4: PDFの構造を解析中...")
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

                # 2. 音声処理
                status_text.text("Step 2/4: オーディオエンジンを起動中...")
                audio_ext = os.path.splitext(uploaded_audio.name)[1]
                audio_path = os.path.join(tmpdir, f"input_audio{audio_ext}")
                with open(audio_path, "wb") as f:
                    f.write(uploaded_audio.read())
                progress_bar.progress(50)

                # 3. AI分析（Whisper）
                status_text.text("Step 3/4: AIが音声を聴き取って同期ポイントを特定中...")
                model = whisper.load_model("base", device="cpu")
                result = model.transcribe(audio_path, language="ja", fp16=False)

                markers = [{"slide": 1, "start": 0.0}]
                found_slides = {1}

                # キーワード・マッチング（「〇枚目」など）
                for segment in result['segments']:
                    text = segment['text']
                    match = re.search(r"(\d+)\s*(枚目|ページ|スライド)", text)
                    if match:
                        num = int(match.group(1))
                        if num not in found_slides and num <= total_slides:
                            markers.append({"slide": num, "start": segment['start']})
                            found_slides.add(num)

                # タイトル・インテリジェント・補完（欠番対策）
                for page_num, title in slide_titles.items():
                    if page_num not in found_slides and len(title) > 3:
                        for segment in result['segments']:
                            if title in segment['text']:
                                markers.append({"slide": page_num, "start": segment['start']})
                                found_slides.add(page_num)
                                st.write(f"✨ タイトル一致で欠番を補完しました: '{title}' (Slide {page_num})")
                                break
                
                # --- エラー通知機能 ---
                missing_slides = [i for i in range(1, total_slides + 1) if i not in found_slides]
                if missing_slides:
                    st.warning(f"⚠️ 欠落検知: スライド {missing_slides} が特定できずスキップされました。プロンプトを見直すと改善する場合があります。")
                else:
                    st.success("✨ すべてのスライドが完璧に同期されました！")

                progress_bar.progress(75)

                # 4. ビデオ・レンダリング
                status_text.text("Step 4/4: 動画をレンダリング中...")
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
                    status_text.text("動画の生成が完了しました。")
                    st.success("✅ 動画の合成に成功しました！")
                    
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
                    st.error("エラー: スライドの切り替えポイントを1つも特定できませんでした。プロンプトを確認してください。")

        except Exception as e:
            st.error(f"システムエラーが発生しました: {e}")
        finally:
            tmpdir_obj.cleanup()
