import os
import uuid
import gradio as gr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from config import Config
from video_processor import VideoProcessor
from audio_processor import AudioProcessor
from transcription import transcribe_clips, ask_question
from pipeline_utils import assign_speakers, extract_top_speaker_clips, cleanup_temp_files

def run_pipeline(
    video_path,
    reference_image_path,
    hf_token,
    pyannotate_token,
    repo_id,
    device="cpu",
    progress=gr.Progress()
):
    """Main pipeline execution"""
    if not video_path:
        return "Please upload a video file.", None, [], [], {}

    if not reference_image_path:
        return "Please upload a reference face image.", None, [], [], {}

    if not hf_token or not pyannotate_token or not repo_id:
        return "Please provide all required API credentials.", None, [], [], {}

    session_id = str(uuid.uuid4())[:8]
    output_base = f"outputs_{session_id}"
    os.makedirs(output_base, exist_ok=True)

    status_messages = []
    def status_callback(msg):
        status_messages.append(msg)
        progress(len(status_messages)/12, desc=msg)

    video_processor = None
    audio_processor = None

    try:
        progress(0, desc="Initializing processors...")
        video_processor = VideoProcessor(status_callback, device=device)
        audio_processor = AudioProcessor(status_callback)

        progress(0.1, desc="Matching faces...")
        matched_frames_dir, intervals_file = video_processor.match_faces(
            video_path,
            reference_image_path,
            output_base
        )

        progress(0.3, desc="Extracting audio...")
        audio_dir = os.path.join(output_base, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = video_processor.extract_audio(video_path, audio_dir)

        progress(0.4, desc="Performing speaker diarization...")
        diarization_file = audio_processor.diarize_audio(
            audio_path,
            hf_token,
            pyannotate_token,
            repo_id
        )

        progress(0.6, desc="Analyzing lip motion...")
        lip_dir = os.path.join(output_base, "lip_analysis")
        lip_scores_file, lip_scores_df = video_processor.analyze_lip_motion(
            reference_image_path,
            matched_frames_dir,
            audio_path,
            lip_dir
        )

        progress(0.75, desc="Assigning speakers...")
        matched_speakers_file = assign_speakers(intervals_file, diarization_file)

        progress(0.8, desc="Extracting speaker clips...")
        clips_dir = os.path.join(output_base, "clips")
        clip_paths, primary_speaker = extract_top_speaker_clips(
            video_path,
            lip_scores_file,
            matched_speakers_file,
            clips_dir
        )

        progress(0.9, desc="Transcribing clips...")
        transcripts_dir = os.path.join(output_base, "transcripts")
        transcripts = transcribe_clips(clip_paths, transcripts_dir, device=device)

        progress(0.95, desc="Generating visualizations...")
        plot_path = generate_visualization(lip_scores_df, output_base)

        matched_images = []
        for f in sorted(os.listdir(matched_frames_dir))[:5]:
            img_path = os.path.join(matched_frames_dir, f)
            if os.path.exists(img_path):
                matched_images.append(img_path)

        output_files = clip_paths + [
            lip_scores_file,
            matched_speakers_file,
            diarization_file,
            plot_path
        ]

        progress(1.0, desc="Complete!")

        success_msg = (
            f"Pipeline completed successfully!\n\n"
            f"Primary Speaker: {primary_speaker}\n"
            f"Generated {len(clip_paths)} video clips\n"
            f"Transcribed {len(transcripts)} clips"
        )

        return (
            success_msg,
            plot_path,
            matched_images,
            output_files,
            transcripts
        )

    except Exception as e:
        import traceback
        error_msg = f"Pipeline failed: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, None, [], [], {}

    finally:
        if video_processor:
            video_processor.cleanup()

def generate_visualization(lip_scores_df, output_base):
    """Generate speech confidence visualization"""
    try:
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(lip_scores_df["timestamp"], lip_scores_df["lip_score"], 'b-', label="Lip Motion", alpha=0.7)
        plt.plot(lip_scores_df["timestamp"], lip_scores_df["sync_confidence"], 'g-', label="Sync Confidence", alpha=0.7)
        plt.title("Lip Motion and Sync Confidence Scores", fontsize=12, pad=10)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        config = Config()
        smoothed_scores = gaussian_filter1d(lip_scores_df["composite_score"], sigma=config.GAUSSIAN_SIGMA)
        plt.plot(lip_scores_df["timestamp"], lip_scores_df["composite_score"], 'r-', alpha=0.3, label="Raw")
        plt.plot(lip_scores_df["timestamp"], smoothed_scores, 'r-', label="Smoothed", linewidth=2)
        plt.title("Composite Speech Confidence", fontsize=12, pad=10)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = os.path.join(output_base, "speech_confidence.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        return plot_path
    except Exception as e:
        print(f"[WARNING] Visualization generation failed: {str(e)}")
        return None

def chat_handler(message, history, transcripts_state, hf_token):
    """Handle chat interactions with proper history format"""
    if not transcripts_state:
        return history + [[message, "Please run the pipeline first to generate transcripts."]]

    if not message or not message.strip():
        return history + [[message, "Please ask a question."]]

    response = ask_question(transcripts_state, message, hf_token)

    return history + [[message, response]]

with gr.Blocks(theme=gr.themes.Soft(), title="SpeakSeek") as app:
    gr.Markdown("# SpeakSeek")
    gr.Markdown("Identify speakers using face recognition, lip motion, and audio-visual sync")

    transcripts_state = gr.State({})

    with gr.Tabs():
        with gr.TabItem("Pipeline Analysis"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Input Configuration")
                    video_input = gr.Video(label="Upload Video")
                    reference_image = gr.Image(label="Reference Face Image", type="filepath")

                    with gr.Accordion("API Configuration", open=True):
                        hf_token = gr.Textbox(label="Hugging Face Token", type="password",
                                            placeholder="hf_...")
                        pyannotate_token = gr.Textbox(label="PyAnnotate API Key", type="password",
                                                     placeholder="Your PyAnnotate key")
                        repo_id = gr.Textbox(label="HF Repo ID",
                                           placeholder="username/repo-name")

                    device_selector = gr.Radio(
                        choices=["cpu", "cuda"],
                        value="cpu",
                        label="Processing Device"
                    )

                    run_btn = gr.Button("Run Analysis", variant="primary", size="lg")

                with gr.Column():
                    status_output = gr.Textbox(label="Status", interactive=False, lines=6)
                    plot_output = gr.Image(label="Speech Confidence Analysis")

            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Matched Frames Preview", open=False):
                        image_gallery = gr.Gallery(label="Top Matches", columns=5, height=300)

                with gr.Column():
                    with gr.Accordion("Output Files", open=True):
                        clip_outputs = gr.Files(label="Results")

        with gr.TabItem("Chat with Transcripts"):
            gr.Markdown("### Ask questions about the extracted video clips")
            gr.Markdown("*Run the pipeline first to enable chat functionality*")

            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
                type="messages"
            )

            with gr.Row():
                question_input = gr.Textbox(
                    label="Your question",
                    placeholder="Where does the speaker talk about...?",
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    "What did the speaker say about deadlines?",
                    "Summarize the main points discussed.",
                    "Which clip contains discussion about the project timeline?",
                    "What is the speaker's opinion in these clips?"
                ],
                inputs=question_input
            )

    run_btn.click(
        fn=run_pipeline,
        inputs=[video_input, reference_image, hf_token, pyannotate_token, repo_id, device_selector],
        outputs=[status_output, plot_output, image_gallery, clip_outputs, transcripts_state]
    )

    submit_btn.click(
        fn=chat_handler,
        inputs=[question_input, chatbot, transcripts_state, hf_token],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[question_input]
    )

    question_input.submit(
        fn=chat_handler,
        inputs=[question_input, chatbot, transcripts_state, hf_token],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[question_input]
    )

if __name__ == "__main__":
    print("[INFO] Starting SpeakSeek application...")
    print("[INFO] Make sure to set your API credentials in the interface")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
